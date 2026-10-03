import copy
import json
import math
import os

import numpy as np
import pytest
import torch
from accelerate import init_empty_weights
from safetensors.torch import save_file, save_model
from transformers.generation.logits_process import (
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from config.configuration_vibevoice import (
    DEFAULT_ASR_CONFIG,
    QwenConfig,
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceASRConfig,
    VibeVoiceSemanticTokenizerConfig,
)
from vibevoice.modular.modeling_vibevoice_asr_inference import (
    IGNORED_CHECKPOINT_PREFIXES,
    VibeVoiceASRForConditionalInference,
)
from vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceSemanticTokenizerModel,
    VibeVoiceTokenizerStreamingCache,
)
from vibevoice.processor.vibevoice_asr_processor import SYSTEM_PROMPT, VibeVoiceASRProcessor

# The full 8B ASR model is never loaded here: only its config/index metadata and tiny random models.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASR_MODEL_DIR = os.path.join(REPO_ROOT, "models", "VibeVoice-ASR")
ASR_CONFIG_FILE = os.path.join(ASR_MODEL_DIR, "config.json")
ASR_INDEX_FILE = os.path.join(ASR_MODEL_DIR, "model.safetensors.index.json")

TINY_HOP = 4  # product of the tiny encoder ratios
SPEECH_PAD = 5
EOS = 63

TINY_CONFIG = {
    "acoustic_tokenizer_config": {"vae_dim": 8, "encoder_n_filters": 4, "decoder_n_filters": 4,
                                  "encoder_ratios": [2, 2], "encoder_depths": "1-1-1", "fix_std": 0.5,
                                  "std_dist_type": "gaussian"},
    "semantic_tokenizer_config": {"vae_dim": 6, "encoder_n_filters": 4, "encoder_ratios": [2, 2],
                                  "encoder_depths": "1-1-1", "std_dist_type": "none"},
    "decoder_config": {"hidden_size": 32, "intermediate_size": 64, "max_position_embeddings": 512, "model_type": "qwen2",
                       "num_attention_heads": 4, "num_hidden_layers": 3, "num_key_value_heads": 2, "vocab_size": 64,
                       "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "eos_token_id": EOS},
    "diffusion_head_config": {"hidden_size": 32, "head_layers": 1, "latent_size": 8},
}


def tiny_config(**overrides) -> VibeVoiceASRConfig:
    config_dict = copy.deepcopy(TINY_CONFIG)
    config_dict["decoder_config"].update(overrides)
    return VibeVoiceASRConfig.from_dict(config_dict, torch_dtype=torch.float32)


def build_tiny_model(**overrides) -> VibeVoiceASRForConditionalInference:
    torch.manual_seed(0)
    model = VibeVoiceASRForConditionalInference(tiny_config(**overrides))
    # AutoCast layers skip parameter init, so every weight is set explicitly.
    for name, param in model.named_parameters():
        if "norm" in name and name.endswith("weight"):
            torch.nn.init.normal_(param, mean=1.0, std=0.1)
        else:
            torch.nn.init.normal_(param, std=0.2)
    model.device = torch.device("cpu")
    return model.eval()


def make_inputs(num_samples: int = 401, prefix: int = 4, suffix: int = 5):
    num_speech = math.ceil(num_samples / TINY_HOP)
    ids = list(range(1, prefix + 1)) + [SPEECH_PAD] * num_speech + list(range(6, 6 + suffix))
    input_ids = torch.tensor([ids])
    torch.manual_seed(1)
    return {
        "input_ids": input_ids,
        "acoustic_input_mask": input_ids == SPEECH_PAD,
        "speech_tensors": torch.randn(1, num_samples) * 0.3,
        "speech_masks": torch.ones(1, num_speech, dtype=torch.bool),
    }


def run_generate(model, inputs=None, seed=7, **kwargs):
    inputs = inputs or make_inputs()
    kwargs.setdefault("max_new_tokens", 12)
    torch.manual_seed(seed)
    return model.generate(**inputs, show_progress_bar=False, **kwargs)


@pytest.fixture(scope="module")
def processor():
    return VibeVoiceASRProcessor.from_pretrained()


class TestASRConfig:
    @pytest.mark.skipif(not os.path.exists(ASR_CONFIG_FILE), reason="VibeVoice-ASR config not present")
    def test_parses_real_config(self):
        with open(ASR_CONFIG_FILE) as f:
            config = VibeVoiceASRConfig.from_dict(json.load(f), torch_dtype=torch.bfloat16)
        assert not hasattr(config, "diffusion_head_config")
        assert isinstance(config.decoder_config, QwenConfig)
        assert isinstance(config.acoustic_tokenizer_config, VibeVoiceAcousticTokenizerConfig)
        assert isinstance(config.semantic_tokenizer_config, VibeVoiceSemanticTokenizerConfig)
        assert (config.acoustic_vae_dim, config.semantic_vae_dim) == (64, 128)
        assert config.acoustic_tokenizer_config.std_dist_type == "gaussian"
        assert config.semantic_tokenizer_config.std_dist_type == "none"
        assert config.decoder_config.hidden_size == 3584
        assert config.decoder_config.num_hidden_layers == 28
        assert config.decoder_config.max_position_embeddings == 131072
        assert config.decoder_config.eos_token_id == 151643
        assert config.torch_dtype == torch.bfloat16

    @pytest.mark.skipif(not os.path.exists(ASR_CONFIG_FILE), reason="VibeVoice-ASR config not present")
    def test_default_config_matches_released_config(self):
        with open(ASR_CONFIG_FILE) as f:
            released = json.load(f)
        released.pop("diffusion_head_config")
        assert DEFAULT_ASR_CONFIG == released

        from_file = VibeVoiceASRConfig.from_dict(released, torch_dtype=torch.bfloat16)
        from_default = VibeVoiceASRConfig.from_dict(DEFAULT_ASR_CONFIG, torch_dtype=torch.bfloat16)
        for name in ("acoustic_tokenizer_config", "semantic_tokenizer_config", "decoder_config"):
            assert vars(getattr(from_default, name)) == vars(getattr(from_file, name))
        assert (from_default.acoustic_vae_dim, from_default.semantic_vae_dim) == (64, 128)

    def test_default_config_not_mutated_by_parsing(self):
        snapshot = copy.deepcopy(DEFAULT_ASR_CONFIG)
        VibeVoiceASRConfig.from_dict(DEFAULT_ASR_CONFIG, torch_dtype=torch.float32)
        VibeVoiceASRConfig.from_dict(DEFAULT_ASR_CONFIG, torch_dtype=torch.bfloat16)
        assert DEFAULT_ASR_CONFIG == snapshot

    def test_vae_dims_fall_back_to_sub_configs(self):
        config = tiny_config()
        assert (config.acoustic_vae_dim, config.semantic_vae_dim) == (8, 6)

    def test_rejects_unsupported_decoder(self):
        config_dict = copy.deepcopy(TINY_CONFIG)
        config_dict["decoder_config"]["model_type"] = "llama"
        with pytest.raises(ValueError):
            VibeVoiceASRConfig.from_dict(config_dict)


class TestASRTokenizer:
    def test_special_token_ids(self, processor):
        tokenizer = processor.tokenizer
        assert tokenizer.speech_start_id == 151646
        assert tokenizer.speech_end_id == 151647
        assert tokenizer.speech_pad_id == 151648
        assert tokenizer.pad_id == 151655
        assert tokenizer.eos_token_id == tokenizer.eos_id == 151643

    def test_chat_template(self, processor):
        tokenizer = processor.tokenizer
        assert tokenizer.apply_chat_template([{"role": "system", "content": "S"}], tokenize=False) == \
            "<|im_start|>system\nS<|im_end|>\n"
        user_ids = tokenizer.apply_chat_template([{"role": "user", "content": "U"}], tokenize=True)
        assert tokenizer.decode(user_ids) == "<|im_start|>user\nU<|im_end|>\n"

    def test_tts_tokenizer_unchanged(self):
        from util import vibevoice_root_dir
        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast

        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
            os.path.join(vibevoice_root_dir, "tokenizer"), local_files_only=True
        )
        assert (tokenizer.speech_start_id, tokenizer.speech_end_id, tokenizer.speech_diffusion_id) == \
            (151652, 151653, 151654)
        assert (tokenizer.eos_id, tokenizer.pad_id) == (151643, 151655)


class TestFinalChunk:
    @pytest.fixture(params=["acoustic", "semantic"])
    def encoder(self, request):
        torch.manual_seed(0)
        if request.param == "acoustic":
            model = VibeVoiceAcousticTokenizerModel(
                VibeVoiceAcousticTokenizerConfig(**TINY_CONFIG["acoustic_tokenizer_config"]), dtype=torch.float32
            )
        else:
            model = VibeVoiceSemanticTokenizerModel(
                VibeVoiceSemanticTokenizerConfig(**TINY_CONFIG["semantic_tokenizer_config"]), dtype=torch.float32
            )
        for param in model.parameters():
            torch.nn.init.normal_(param, std=0.2)
        return model.eval()

    @staticmethod
    def segmented(encoder, audio, bounds, final_flag):
        cache = VibeVoiceTokenizerStreamingCache()
        means = []
        with torch.no_grad():
            for idx, (start, end) in enumerate(bounds):
                means.append(encoder.encode(
                    audio[:, :, start:end], cache=cache, sample_indices=torch.arange(1), use_cache=True,
                    is_final_chunk=final_flag and idx == len(bounds) - 1,
                ).mean)
        return torch.cat(means, dim=1)

    def test_default_is_unchanged(self, encoder):
        audio = torch.randn(1, 1, 203)
        with torch.no_grad():
            assert torch.equal(encoder.encode(audio).mean, encoder.encode(audio, is_final_chunk=False).mean)

    def test_final_chunk_matches_one_shot(self, encoder):
        audio = torch.randn(1, 1, 203)
        bounds = [(0, 80), (80, 160), (160, 203)]
        with torch.no_grad():
            one_shot = encoder.encode(audio).mean
        without_final = self.segmented(encoder, audio, bounds, False)
        with_final = self.segmented(encoder, audio, bounds, True)

        assert one_shot.shape[1] == math.ceil(203 / TINY_HOP)
        assert without_final.shape[1] == 203 // TINY_HOP
        assert with_final.shape == one_shot.shape
        torch.testing.assert_close(with_final, one_shot, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(without_final, one_shot[:, :without_final.shape[1]], atol=1e-5, rtol=1e-5)


class TestASRProcessor:
    def test_prompt_format(self, processor):
        audio = np.random.RandomState(0).randn(24000 * 2 + 100).astype(np.float32) * 0.1
        inputs = processor(audio)
        num_speech = math.ceil(len(audio) / 3200)
        duration = len(audio) / 24000

        expected = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n<|object_ref_start|>{'<|box_start|>' * num_speech}<|object_ref_end|>\n"
            f"This is a {duration:.2f} seconds audio, please transcribe it with these keys: "
            f"Start time, End time, Speaker ID, Content<|im_end|>\n"
        )
        assert processor.decode(inputs["input_ids"][0]) == expected
        assert "<|im_start|>assistant" not in processor.decode(inputs["input_ids"][0])
        assert int(inputs["acoustic_input_mask"].sum()) == num_speech
        assert inputs["speech_masks"].shape == (1, num_speech) and bool(inputs["speech_masks"].all())
        assert inputs["speech_tensors"].shape == (1, len(audio))
        assert inputs["attention_mask"].shape == inputs["input_ids"].shape
        assert inputs["audio_duration"] == pytest.approx(duration)

    def test_context_info_prompt(self, processor):
        audio = np.zeros(32000, dtype=np.float32)
        text = processor.decode(processor(audio, context_info="  Alice, Bob ")["input_ids"][0])
        assert ("This is a 1.33 seconds audio, with extra info: Alice, Bob\n\n"
                "Please transcribe it with these keys: Start time, End time, Speaker ID, Content<|im_end|>\n") in text
        assert "please transcribe" not in processor.decode(processor(audio, context_info="Alice")["input_ids"][0])
        assert processor.decode(processor(audio, context_info="   ")["input_ids"][0]) == \
            processor.decode(processor(audio)["input_ids"][0])

    def test_audio_is_normalized(self, processor):
        audio = np.random.RandomState(1).randn(24000).astype(np.float32) * 0.001
        speech = processor(torch.from_numpy(audio))["speech_tensors"][0].numpy()
        rms_db = 20 * np.log10(np.sqrt(np.mean(speech ** 2)))
        assert rms_db == pytest.approx(-25, abs=0.5)

    @pytest.mark.parametrize("text,expected", [
        ('[{"Start time": 0.0, "End time": 1.5, "Speaker ID": 0, "Content": "hi"}]',
         [{"start_time": 0.0, "end_time": 1.5, "speaker_id": 0, "text": "hi"}]),
        ('prefix ```json\n[{"Start": 1, "End": 2, "Speaker": 1, "Content": "a"}]\n``` suffix',
         [{"start_time": 1, "end_time": 2, "speaker_id": 1, "text": "a"}]),
        ('noise {"Content": "only"} trailing', [{"text": "only"}]),
        ('[{"Unknown": 1}, {"Content": "x"}]', [{"text": "x"}]),
        ("not json at all", []),
        ('[{"Content": "truncated', []),
    ])
    def test_post_process(self, processor, text, expected):
        assert processor.post_process_transcription(text) == expected


class TestASRModelStructure:
    @pytest.mark.skipif(not os.path.exists(ASR_INDEX_FILE), reason="VibeVoice-ASR index not present")
    @pytest.mark.parametrize("source", ["file", "default"])
    def test_state_dict_matches_checkpoint_index(self, source):
        if source == "file":
            with open(ASR_CONFIG_FILE) as f:
                config_dict = json.load(f)
        else:
            config_dict = DEFAULT_ASR_CONFIG
        config = VibeVoiceASRConfig.from_dict(config_dict, torch_dtype=torch.bfloat16)
        with init_empty_weights():
            model = VibeVoiceASRForConditionalInference(config)
        with open(ASR_INDEX_FILE) as f:
            checkpoint_keys = set(json.load(f)["weight_map"])

        used_keys = {k for k in checkpoint_keys if not k.startswith(IGNORED_CHECKPOINT_PREFIXES)}
        assert set(model.state_dict().keys()) == used_keys
        assert len(checkpoint_keys - used_keys) == 276
        assert model.model.acoustic_tokenizer.decoder is None
        assert model.model.language_model.rotary_emb.inv_freq.dtype == torch.float32

    def test_logits_processor_order(self):
        build = VibeVoiceASRForConditionalInference.build_logits_processor
        assert len(build(0.0, 50, 0.9, 1.0)) == 0
        assert [type(p) for p in build(0.0, 50, 0.9, 1.2)] == [RepetitionPenaltyLogitsProcessor]
        assert [type(p) for p in build(0.7, 50, 0.9, 1.2)] == [
            RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
        ]
        assert [type(p) for p in build(1.0, 0, 1.0, 1.0)] == []


class TestASRGenerate:
    def test_greedy_is_deterministic(self):
        model = build_tiny_model()
        first = run_generate(model, seed=1)
        second = run_generate(model, seed=2)
        assert first.generated_ids == second.generated_ids
        assert len(first.generated_ids) == 12 and first.reach_max_new_tokens
        assert first.sequences.shape[1] == first.prompt_tokens + 12
        assert first.speech_tokens == math.ceil(401 / TINY_HOP)

    def test_sampling_depends_on_seed(self):
        model = build_tiny_model()
        a = run_generate(model, seed=3, temperature=1.0, max_new_tokens=20)
        b = run_generate(model, seed=3, temperature=1.0, max_new_tokens=20)
        c = run_generate(model, seed=4, temperature=1.0, max_new_tokens=20)
        assert a.generated_ids == b.generated_ids
        assert a.generated_ids != c.generated_ids

    def test_cache_matches_full_recompute(self):
        model = build_tiny_model()
        inputs = make_inputs()
        step_logits = []
        lm_step = model._lm_step
        model._lm_step = lambda *args: step_logits.append(lm_step(*args)) or step_logits[-1]
        out = run_generate(model, inputs, temperature=1.0, max_new_tokens=8)

        with torch.no_grad():
            torch.manual_seed(7)
            speech = model.encode_speech(inputs["speech_tensors"], inputs["speech_masks"])
            embeds = model.model.language_model.embed_tokens(out.sequences[:, :-1])
            embeds[0, :out.prompt_tokens][inputs["acoustic_input_mask"][0]] = speech
            hidden = model.model.language_model(inputs_embeds=embeds).last_hidden_state
            full_logits = model.lm_head(hidden[0, out.prompt_tokens - 1:]).float()
        torch.testing.assert_close(torch.cat(step_logits), full_logits, atol=1e-4, rtol=1e-4)

    def test_stops_on_eos(self):
        model = build_tiny_model()
        greedy = run_generate(model).generated_ids
        eos = greedy[2]
        out = run_generate(model, eos_token_id=eos)
        assert out.generated_ids == greedy[:greedy.index(eos) + 1]
        assert not out.reach_max_new_tokens
        out = run_generate(model, eos_token_id=[999, greedy[0]])
        assert out.generated_ids == greedy[:1]

    def test_default_eos_from_config(self):
        model = build_tiny_model()
        greedy = run_generate(model).generated_ids
        model.config.decoder_config.eos_token_id = greedy[1]
        assert run_generate(model).generated_ids == greedy[:greedy.index(greedy[1]) + 1]

    def test_stop_check_fn_and_callback(self):
        model = build_tiny_model()
        received = []
        out = run_generate(model, stop_check_fn=lambda: len(received) >= 4, token_callback=received.append)
        assert out.stopped_externally and not out.reach_max_new_tokens
        assert out.generated_ids == received and len(received) == 4

    def test_max_new_tokens_clamped_to_context(self):
        inputs = make_inputs()
        prompt_len = inputs["input_ids"].shape[1]
        model = build_tiny_model(max_position_embeddings=prompt_len + 3)
        out = run_generate(model, inputs, max_new_tokens=100)
        assert len(out.generated_ids) == 3 and out.reach_max_new_tokens

        model = build_tiny_model(max_position_embeddings=prompt_len)
        with pytest.raises(ValueError, match="exceeds the model context"):
            run_generate(model, inputs)

    def test_speech_count_mismatch_raises(self):
        model = build_tiny_model()
        inputs = make_inputs()
        inputs["speech_tensors"] = inputs["speech_tensors"][:, :-TINY_HOP * 2]
        inputs["speech_masks"] = inputs["speech_masks"][:, :-2]
        with pytest.raises(ValueError, match="speech positions"):
            run_generate(model, inputs)

    def test_rejects_batch_greater_than_one(self):
        model = build_tiny_model()
        inputs = make_inputs()
        inputs["input_ids"] = inputs["input_ids"].repeat(2, 1)
        with pytest.raises(AssertionError):
            run_generate(model, inputs)

    def test_segmented_encode_matches_one_shot_length(self):
        model = build_tiny_model()
        # Noise layout differs between the paths (randn_like keeps the one-shot mean's permuted strides), as upstream.
        model.model.acoustic_tokenizer.fix_std.fill_(0.0)
        audio = torch.randn(1, 1003) * 0.3
        mask = torch.ones(1, math.ceil(1003 / TINY_HOP), dtype=torch.bool)
        torch.manual_seed(0)
        one_shot = model.encode_speech(audio, mask)
        torch.manual_seed(0)
        segmented = model.encode_speech(audio, mask, streaming_segment_duration=400 / 24000)
        assert segmented.shape == one_shot.shape == (mask.shape[1], 32)
        torch.testing.assert_close(segmented, one_shot, atol=1e-4, rtol=1e-4)


class TestASRLoading:
    @staticmethod
    def save_tiny_checkpoint(path, extra=None):
        model = build_tiny_model()
        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
        state_dict["model.acoustic_tokenizer.decoder.fake.weight"] = torch.zeros(2)
        state_dict.update(extra or {})
        save_file(state_dict, str(path))
        return model

    def test_mono_file_ignores_decoder_and_keeps_rope_fp32(self, tmp_path):
        path = tmp_path / "model.safetensors"
        reference = self.save_tiny_checkpoint(path)

        loaded = VibeVoiceASRForConditionalInference.from_pretrain(str(tmp_path), tiny_config(), device="cpu").eval()
        assert loaded.dtype == torch.float32
        assert run_generate(loaded).generated_ids == run_generate(reference).generated_ids

        loaded = VibeVoiceASRForConditionalInference.from_pretrain(
            str(path), tiny_config(), device="cpu", dtype=torch.bfloat16,
        ).eval()
        assert loaded.dtype == loaded.compute_dtype == torch.bfloat16
        assert loaded.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16
        assert loaded.model.language_model.rotary_emb.inv_freq.dtype == torch.float32
        assert len(run_generate(loaded, max_new_tokens=3).generated_ids) == 3

    def test_unexpected_key_raises(self, tmp_path):
        path = tmp_path / "model.safetensors"
        self.save_tiny_checkpoint(path, extra={"model.unknown.weight": torch.zeros(2)})
        with pytest.raises(RuntimeError, match="Checkpoint mismatch"):
            VibeVoiceASRForConditionalInference.from_pretrain(str(path), tiny_config(), device="cpu")

    def test_converter_round_trip(self, tmp_path):
        source = tmp_path / "model.safetensors"
        self.save_tiny_checkpoint(source)
        model = VibeVoiceASRForConditionalInference.from_pretrain(str(tmp_path), tiny_config(), device="cpu").eval()
        model.to(dtype=torch.float8_e4m3fn)
        converted = tmp_path / "tiny_asr_float8_e4m3fn.safetensors"
        save_model(model, str(converted))

        loaded = VibeVoiceASRForConditionalInference.from_pretrain(str(converted), tiny_config(), device="cpu").eval()
        assert loaded.lm_head.weight.dtype == torch.float8_e4m3fn
        assert loaded.compute_dtype == torch.bfloat16
        assert loaded.model.language_model.rotary_emb.inv_freq.dtype == torch.float32
