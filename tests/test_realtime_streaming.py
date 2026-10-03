import copy
import json
import os

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from config.configuration_vibevoice import DEFAULT_CONFIG, QwenConfig, VibeVoiceConfig, VibeVoiceStreamingConfig
from util.rand_init import get_generator
from util.streaming_voice_preset import PRESET_GROUPS, convert_pt_voice_preset, load_voice_preset
from vibevoice.modular.modular_vibevoice_qwen import QwenModel
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    StreamingQwenModel,
    VibeVoiceStreamingForConditionalInference,
)
from vibevoice.modular.streamer import AudioStreamer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REALTIME_MODEL_DIR = os.path.join(REPO_ROOT, "models", "VibeVoice-Realtime-0.5B")
REALTIME_MODEL_FILE = os.path.join(REALTIME_MODEL_DIR, "model.safetensors")
CARTER_PRESET = os.path.join(REPO_ROOT, "demo", "voices", "streaming_model", "en-Carter_man.safetensors")

REALTIME_CONFIG = {
    "acoustic_vae_dim": 64,
    "acoustic_tokenizer_config": {"vae_dim": 64, "decoder_ratios": [8, 5, 5, 4, 2, 2], "encoder_depths": "3-3-3-3-3-3-8"},
    "decoder_config": {
        "hidden_size": 896, "intermediate_size": 4864, "max_position_embeddings": 8192, "model_type": "qwen2",
        "num_attention_heads": 14, "num_hidden_layers": 24, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0, "tie_word_embeddings": False, "vocab_size": 151936,
    },
    "diffusion_head_config": {"hidden_size": 896, "head_layers": 4, "latent_size": 64, "prediction_type": "v_prediction",
                              "ddpm_beta_schedule": "cosine", "ddpm_num_steps": 1000, "ddpm_num_inference_steps": 20},
    "model_type": "vibevoice_streaming",
    "torch_dtype": "bfloat16",
    "tts_backbone_num_hidden_layers": 20,
}

TINY_HOP = 4  # product of the tiny decoder ratios


def tiny_config(**kwargs) -> VibeVoiceStreamingConfig:
    config_dict = {
        "acoustic_tokenizer_config": {
            "vae_dim": 8, "encoder_n_filters": 4, "decoder_n_filters": 4,
            "encoder_ratios": [2, 2], "encoder_depths": "1-1-1",
        },
        "decoder_config": {
            "hidden_size": 32, "intermediate_size": 64, "max_position_embeddings": 512, "model_type": "qwen2",
            "num_attention_heads": 4, "num_hidden_layers": 3, "num_key_value_heads": 2, "vocab_size": 64,
        },
        "diffusion_head_config": {"hidden_size": 32, "head_layers": 1, "latent_size": 8, "ddpm_num_inference_steps": 3},
        "tts_backbone_num_hidden_layers": 2,
    }
    return VibeVoiceStreamingConfig.from_dict(config_dict, torch_dtype=torch.float32, **kwargs)


def build_tiny_model(eos_bias: float = -10.0) -> VibeVoiceStreamingForConditionalInference:
    torch.manual_seed(0)
    model = VibeVoiceStreamingForConditionalInference(tiny_config())
    # AutoCast layers skip parameter init, so every weight is set explicitly.
    for name, param in model.named_parameters():
        if "norm" in name and name.endswith("weight"):
            torch.nn.init.ones_(param)
        else:
            torch.nn.init.normal_(param, std=0.05)
    torch.nn.init.constant_(model.tts_eos_classifier.fc2.bias, eos_bias)
    model.model.speech_scaling_factor.fill_(1.0)
    model.model.speech_bias_factor.fill_(0.0)
    model.device = torch.device("cpu")
    return model.eval()


def build_prefilled(model, lm_len: int = 7, tts_len: int = 9):
    hidden = model.config.decoder_config.hidden_size
    lm = model.model.language_model(inputs_embeds=torch.randn(1, lm_len, hidden), past_key_values=DynamicCache())
    tts_lm = model.model.tts_language_model(inputs_embeds=torch.randn(1, tts_len, hidden), past_key_values=DynamicCache())
    neg_lm = model.model.language_model(inputs_embeds=torch.randn(1, 1, hidden), past_key_values=DynamicCache())
    neg_tts_lm = model.model.tts_language_model(inputs_embeds=torch.randn(1, 1, hidden), past_key_values=DynamicCache())
    return {"lm": lm, "tts_lm": tts_lm, "neg_lm": neg_lm, "neg_tts_lm": neg_tts_lm}


def run_generate(model, prefilled, text_len=12, seed=7, **kwargs):
    get_generator(seed, force_set=True)
    tts_text_ids = torch.arange(1, text_len + 1).unsqueeze(0)
    return model.generate(tts_text_ids=tts_text_ids, all_prefilled_outputs=copy.deepcopy(prefilled),
                          show_progress_bar=False, **kwargs)


class TestStreamingConfig:
    def test_from_dict_parses_realtime_config(self):
        config = VibeVoiceStreamingConfig.from_dict(copy.deepcopy(REALTIME_CONFIG), torch_dtype=torch.float32)
        assert isinstance(config.decoder_config, QwenConfig)
        assert config.decoder_config.hidden_size == 896
        assert config.decoder_config.num_hidden_layers == 24
        assert config.tts_backbone_num_hidden_layers == 20
        assert config.acoustic_vae_dim == 64
        assert config.diffusion_head_config.prediction_type == "v_prediction"
        assert config.torch_dtype == torch.float32
        assert config.model_type == "vibevoice_streaming"

    def test_rejects_unsupported_decoder(self):
        config_dict = copy.deepcopy(REALTIME_CONFIG)
        config_dict["decoder_config"]["model_type"] = "llama"
        with pytest.raises(ValueError):
            VibeVoiceStreamingConfig.from_dict(config_dict)

    def test_existing_config_unaffected(self):
        config = VibeVoiceConfig.from_dict(copy.deepcopy(DEFAULT_CONFIG), torch_dtype=torch.bfloat16)
        assert config.decoder_config.num_hidden_layers == 28
        assert config.semantic_vae_dim == 128
        assert not hasattr(config, "tts_backbone_num_hidden_layers")


class TestStreamingQwenMask:
    @staticmethod
    def _models():
        config = tiny_config().decoder_config
        torch.manual_seed(0)
        streaming = StreamingQwenModel(config, dtype=torch.float32)
        for param in streaming.parameters():
            torch.nn.init.normal_(param, std=0.1)
        base = QwenModel(config, dtype=torch.float32)
        base.load_state_dict(streaming.state_dict())
        return streaming, base

    @staticmethod
    def _windowed_vs_stepwise(model, prompt, window):
        cache_a = DynamicCache()
        model(inputs_embeds=prompt, past_key_values=cache_a)
        windowed = model(inputs_embeds=window, past_key_values=cache_a).last_hidden_state

        cache_b = DynamicCache()
        model(inputs_embeds=prompt, past_key_values=cache_b)
        stepwise = torch.cat([
            model(inputs_embeds=window[:, i:i + 1], past_key_values=cache_b).last_hidden_state
            for i in range(window.shape[1])
        ], dim=1)
        return windowed, stepwise

    def test_window_on_cache_matches_token_by_token(self):
        streaming, _ = self._models()
        prompt, window = torch.randn(1, 6, 32), torch.randn(1, 5, 32)
        windowed, stepwise = self._windowed_vs_stepwise(streaming, prompt, window)
        torch.testing.assert_close(windowed, stepwise, atol=1e-5, rtol=1e-5)

    def test_base_qwen_model_would_be_wrong(self):
        """Guards the reason StreamingQwenModel exists: SDPA is_causal aligns top-left."""
        _, base = self._models()
        prompt, window = torch.randn(1, 6, 32), torch.randn(1, 5, 32)
        windowed, stepwise = self._windowed_vs_stepwise(base, prompt, window)
        assert not torch.allclose(windowed, stepwise, atol=1e-4)

    def test_prefill_without_cache_matches_base(self):
        streaming, base = self._models()
        x = torch.randn(1, 8, 32)
        torch.testing.assert_close(
            streaming(inputs_embeds=x, past_key_values=DynamicCache()).last_hidden_state,
            base(inputs_embeds=x, past_key_values=DynamicCache()).last_hidden_state,
        )


class TestStreamingModelStructure:
    def test_layer_split_and_decoder_only_tokenizer(self):
        from accelerate import init_empty_weights
        config = VibeVoiceStreamingConfig.from_dict(copy.deepcopy(REALTIME_CONFIG), torch_dtype=torch.bfloat16)
        with init_empty_weights():
            model = VibeVoiceStreamingForConditionalInference(config)
        assert len(model.model.language_model.layers) == 4
        assert len(model.model.tts_language_model.layers) == 20
        assert isinstance(model.model.language_model.norm, torch.nn.Identity)
        assert model.model.acoustic_tokenizer.encoder is None
        assert model.tts_eos_classifier.fc2.out_features == 1

    @pytest.mark.skipif(not os.path.exists(REALTIME_MODEL_FILE), reason="Realtime 0.5B checkpoint not downloaded")
    def test_state_dict_matches_checkpoint(self):
        from accelerate import init_empty_weights
        from safetensors import safe_open
        with open(os.path.join(REALTIME_MODEL_DIR, "config.json")) as f:
            config = VibeVoiceStreamingConfig.from_dict(json.load(f), torch_dtype=torch.bfloat16)
        with init_empty_weights():
            model = VibeVoiceStreamingForConditionalInference(config)
        model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        with safe_open(REALTIME_MODEL_FILE, framework="pt") as f:
            ckpt_shapes = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
        assert set(model_shapes) == set(ckpt_shapes)
        assert model_shapes == ckpt_shapes


class TestVoicePreset:
    @staticmethod
    def _fake_prefilled(num_layers=(2, 3, 2, 3), lengths=(5, 7, 1, 1)):
        prefilled = {}
        for group, layers, length in zip(PRESET_GROUPS, num_layers, lengths):
            cache = DynamicCache()
            for idx in range(layers):
                cache.update(torch.randn(1, 2, length, 4, dtype=torch.bfloat16),
                             torch.randn(1, 2, length, 4, dtype=torch.bfloat16), idx)
            prefilled[group] = BaseModelOutputWithPast(
                last_hidden_state=torch.randn(1, length, 8, dtype=torch.bfloat16), past_key_values=cache,
            )
        return prefilled

    def test_round_trip(self, tmp_path):
        prefilled = self._fake_prefilled()
        pt_path, st_path = tmp_path / "voice.pt", tmp_path / "voice.safetensors"
        torch.save(prefilled, pt_path)
        convert_pt_voice_preset(str(pt_path), str(st_path))

        loaded = load_voice_preset(str(st_path))
        for group in PRESET_GROUPS:
            src, dst = prefilled[group], loaded[group]
            assert torch.equal(src.last_hidden_state, dst.last_hidden_state)
            assert dst.past_key_values.get_seq_length() == src.past_key_values.get_seq_length()
            assert len(dst.past_key_values) == len(src.past_key_values)
            for idx in range(len(src.past_key_values)):
                assert torch.equal(src.past_key_values.key_cache[idx], dst.past_key_values.key_cache[idx])
                assert torch.equal(src.past_key_values.value_cache[idx], dst.past_key_values.value_cache[idx])
            assert dst["last_hidden_state"] is dst.last_hidden_state

    def test_dtype_cast(self, tmp_path):
        pt_path, st_path = tmp_path / "voice.pt", tmp_path / "voice.safetensors"
        torch.save(self._fake_prefilled(), pt_path)
        convert_pt_voice_preset(str(pt_path), str(st_path))
        loaded = load_voice_preset(str(st_path), dtype=torch.float32)
        assert loaded["tts_lm"].last_hidden_state.dtype == torch.float32
        assert loaded["tts_lm"].past_key_values.key_cache[0].dtype == torch.float32

    def test_rejects_other_safetensors(self, tmp_path):
        from safetensors.torch import save_file
        path = tmp_path / "model.safetensors"
        save_file({"x": torch.zeros(1)}, str(path))
        with pytest.raises(ValueError):
            load_voice_preset(str(path))

    @pytest.mark.skipif(not os.path.exists(CARTER_PRESET), reason="Converted Carter preset not available")
    def test_converted_upstream_preset_layout(self):
        loaded = load_voice_preset(CARTER_PRESET)
        assert len(loaded["lm"].past_key_values) == 4
        assert len(loaded["tts_lm"].past_key_values) == 20
        assert len(loaded["neg_tts_lm"].past_key_values) == 20
        for group in PRESET_GROUPS:
            output = loaded[group]
            assert output.past_key_values.get_seq_length() == output.last_hidden_state.shape[1]


class TestStreamingProcessor:
    @pytest.fixture(scope="class")
    def processor(self):
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        return VibeVoiceStreamingProcessor.from_pretrained(None)

    def test_process_input_with_cached_prompt(self, processor):
        cached_prompt = {
            "lm": {"last_hidden_state": torch.zeros(1, 7, 8)},
            "tts_lm": {"last_hidden_state": torch.zeros(1, 11, 8)},
        }
        text = "  Hello world.  "
        inputs = processor.process_input_with_cached_prompt(text=text, cached_prompt=cached_prompt)

        expected_text_ids = processor.tokenizer.encode("Hello world.\n", add_special_tokens=False)
        assert inputs["tts_text_ids"].tolist() == [expected_text_ids]
        assert inputs["input_ids"].shape == (1, 7)
        assert inputs["tts_lm_input_ids"].shape == (1, 11)
        assert (inputs["tts_lm_input_ids"] == processor.tokenizer.pad_id).all()
        assert inputs["tts_lm_attention_mask"].sum().item() == 11
        assert not inputs["speech_input_mask"].any()
        assert inputs["speech_tensors"] is None

    def test_tokenizer_ids_fit_realtime_vocab(self, processor):
        tokenizer = processor.tokenizer
        assert tokenizer.convert_tokens_to_ids("<|image_pad|>") == tokenizer.pad_id
        assert len(tokenizer) <= REALTIME_CONFIG["decoder_config"]["vocab_size"]


class TestStreamingGenerate:
    def test_stops_at_max_length(self):
        model = build_tiny_model(eos_bias=-10.0)
        prefilled = build_prefilled(model)
        output = run_generate(model, prefilled, max_new_tokens=20)
        audio = output.speech_outputs[0]
        assert output.reach_max_step_sample.item()
        assert audio is not None and audio.shape[-1] % TINY_HOP == 0
        assert output.sequences.shape[1] == 9 + 20 + 1

    def test_stops_on_eos(self):
        model = build_tiny_model(eos_bias=10.0)
        prefilled = build_prefilled(model)
        output = run_generate(model, prefilled, max_new_tokens=50)
        assert not output.reach_max_step_sample.item()
        assert output.speech_outputs[0].shape[-1] == TINY_HOP
        assert output.sequences.shape[1] == 9 + 5 + 1

    def test_stop_check_fn(self):
        model = build_tiny_model()
        output = run_generate(model, build_prefilled(model), stop_check_fn=lambda: True)
        assert output.speech_outputs == [None]

    def test_deterministic_with_seed(self):
        model = build_tiny_model()
        prefilled = build_prefilled(model)
        first = run_generate(model, prefilled, max_new_tokens=15).speech_outputs[0]
        second = run_generate(model, prefilled, max_new_tokens=15).speech_outputs[0]
        third = run_generate(model, prefilled, max_new_tokens=15, seed=8).speech_outputs[0]
        assert torch.equal(first, second)
        assert not torch.equal(first, third)

    def test_audio_streamer_receives_all_chunks(self):
        model = build_tiny_model()
        streamer = AudioStreamer(batch_size=1)
        output = run_generate(model, build_prefilled(model), max_new_tokens=15, audio_streamer=streamer)
        streamed = torch.cat(list(streamer.get_stream(0)), dim=-1)
        torch.testing.assert_close(streamed, output.speech_outputs[0])

    def test_prefilled_caches_mutated_in_place(self):
        model = build_tiny_model()
        prefilled = build_prefilled(model)
        get_generator(7, force_set=True)
        model.generate(tts_text_ids=torch.arange(1, 4).unsqueeze(0), all_prefilled_outputs=prefilled,
                       max_new_tokens=10, show_progress_bar=False)
        assert prefilled["tts_lm"].past_key_values.get_seq_length() > 9


@pytest.mark.skipif(
    os.environ.get("VIBEVOICE_RUN_MODEL_TESTS") != "1" or not os.path.exists(REALTIME_MODEL_FILE)
    or not os.path.exists(CARTER_PRESET),
    reason="Set VIBEVOICE_RUN_MODEL_TESTS=1 with the checkpoint and converted presets to run",
)
class TestRealtimeModelCPU:
    def test_generate_short_text(self):
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        with open(os.path.join(REALTIME_MODEL_DIR, "config.json")) as f:
            config = VibeVoiceStreamingConfig.from_dict(json.load(f), torch_dtype=torch.float32)
        model = VibeVoiceStreamingForConditionalInference.from_pretrain(
            REALTIME_MODEL_DIR, config, device="cpu", dtype=torch.float32,
        ).eval()
        model.set_ddpm_inference_steps(5)
        processor = VibeVoiceStreamingProcessor.from_pretrained(REALTIME_MODEL_DIR)
        prefilled = load_voice_preset(CARTER_PRESET, dtype=torch.float32)
        inputs = processor.process_input_with_cached_prompt(text="Hello, this is a test.", cached_prompt=prefilled)

        get_generator(42, force_set=True)
        output = model.generate(tts_text_ids=inputs["tts_text_ids"], tts_lm_input_ids=inputs["tts_lm_input_ids"],
                                all_prefilled_outputs=prefilled, show_progress_bar=False)
        audio = output.speech_outputs[0]
        assert not output.reach_max_step_sample.item()
        assert audio is not None and audio.shape[-1] > 24000 * 0.5
        assert torch.isfinite(audio).all()
