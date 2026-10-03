import argparse
import json
import os

import torch

from config.configuration_vibevoice import DEFAULT_ASR_CONFIG, VibeVoiceASRConfig
from util.rand_init import get_generator
from vibevoice.modular.custom_offloading_utils import OffloadConfig
from vibevoice.modular.modeling_vibevoice_asr_inference import VibeVoiceASRForConditionalInference
from vibevoice.processor.vibevoice_asr_processor import VibeVoiceASRProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice-ASR transcription from audio files")
    parser.add_argument("--model_path", type=str, default="models/VibeVoice-ASR",
                        help="HF model directory (model.safetensors.index.json) or a converted single safetensors file")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json path; defaults to <model_path>/config.json, "
                             "or the built-in ASR config when that file is absent")
    parser.add_argument("--audio_files", type=str, nargs="+", required=True, help="Audio files to transcribe")
    parser.add_argument("--context_info", type=str, default=None,
                        help="Optional extra info (hotwords, names, topic) added to the prompt")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save transcripts")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float32", "float8_e4m3fn"],
                        help="Cast weights on load; defaults to the checkpoint dtype")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling (HF default, used by upstream)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="1.0 = no penalty")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (acoustic latent sampling and decoding)")
    parser.add_argument("--offload_layers_on_gpu", type=int, default=None,
                        help="Enable layer offloading, keeping N of the 28 LM layers on GPU")
    return parser.parse_args()


def main():
    args = parse_args()

    model_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    config_path = args.config or os.path.join(model_dir, "config.json")
    if args.config or os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"{config_path} not found, using default ASR configuration")
        config_dict = DEFAULT_ASR_CONFIG
    dtype = getattr(torch, args.dtype) if args.dtype else None
    config = VibeVoiceASRConfig.from_dict(
        config_dict, torch_dtype=torch.float32 if dtype == torch.float32 else torch.bfloat16
    )

    offload_config = None
    if args.offload_layers_on_gpu is not None:
        offload_config = OffloadConfig(enabled=True, num_layers_on_gpu=args.offload_layers_on_gpu)

    processor = VibeVoiceASRProcessor.from_pretrained(model_dir)
    model = VibeVoiceASRForConditionalInference.from_pretrain(
        args.model_path, config, device=args.device, offload_config=offload_config, dtype=dtype,
    )
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    for audio_file in args.audio_files:
        get_generator(args.seed, force_set=True)
        inputs = processor(audio_file, context_info=args.context_info)
        print(f"\nTranscribing {audio_file} ({inputs['audio_duration']:.2f}s, "
              f"{inputs['input_ids'].shape[1]} prompt tokens)")

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            acoustic_input_mask=inputs["acoustic_input_mask"],
            speech_tensors=inputs["speech_tensors"],
            speech_masks=inputs["speech_masks"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        raw_text = processor.decode(outputs.generated_ids, skip_special_tokens=True)
        segments = processor.post_process_transcription(raw_text)

        result = {
            "file": audio_file,
            "audio_duration": inputs["audio_duration"],
            "raw_text": raw_text,
            "segments": segments,
            "generated_tokens": len(outputs.generated_ids),
            "reach_max_new_tokens": outputs.reach_max_new_tokens,
            "encode_time": outputs.encode_time,
            "prefill_time": outputs.prefill_time,
            "decode_time": outputs.decode_time,
        }
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(os.path.basename(audio_file))[0]}_asr.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print("=" * 50)
        for seg in segments:
            print(f"[{seg.get('start_time', '?')} - {seg.get('end_time', '?')}] "
                  f"Speaker {seg.get('speaker_id', '?')}: {seg.get('text', '')}")
        if not segments:
            print(raw_text)
        print("-" * 50)
        print(f"Output file: {output_path}")
        print(f"Generated tokens: {len(outputs.generated_ids)}"
              f"{' (hit max_new_tokens)' if outputs.reach_max_new_tokens else ''}")
        print(f"Encode / prefill / decode: {outputs.encode_time:.2f}s / {outputs.prefill_time:.2f}s / "
              f"{outputs.decode_time:.2f}s")
        print("=" * 50)


if __name__ == "__main__":
    main()
