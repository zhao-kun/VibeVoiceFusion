import argparse
import glob
import json
import os
import time

import torch

from config.configuration_vibevoice import VibeVoiceStreamingConfig
from util.rand_init import get_generator
from util.streaming_voice_preset import load_voice_preset
from vibevoice.modular.custom_offloading_utils import OffloadConfig
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor


def find_voice_preset(voices_dir: str, speaker_name: str) -> str:
    presets = {
        os.path.splitext(os.path.basename(p))[0].lower(): p
        for p in sorted(glob.glob(os.path.join(voices_dir, "**", "*.safetensors"), recursive=True))
    }
    if not presets:
        raise FileNotFoundError(
            f"No voice presets in {voices_dir}. Convert upstream .pt presets with demo/convert_realtime_voice_presets.py"
        )
    speaker_name = speaker_name.lower()
    if speaker_name in presets:
        return presets[speaker_name]
    matches = [p for name, p in presets.items() if speaker_name in name]
    if len(matches) > 1:
        raise ValueError(f"Multiple voice presets match '{speaker_name}': {matches}")
    if not matches:
        raise ValueError(f"No voice preset matches '{speaker_name}'. Available: {', '.join(presets)}")
    return matches[0]


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice Realtime 0.5B inference from a text file")
    parser.add_argument("--model_path", type=str, default="models/VibeVoice-Realtime-0.5B",
                        help="Model directory (config.json + model.safetensors) or a single safetensors file")
    parser.add_argument("--config", type=str, default=None,
                        help="config.json path, defaults to <model_path>/config.json")
    parser.add_argument("--txt_path", type=str, required=True, help="Text file with the script (plain text)")
    parser.add_argument("--speaker_name", type=str, default="Carter", help="Voice preset name, e.g. Carter or en-Carter_man")
    parser.add_argument("--voices_dir", type=str, default="demo/voices/streaming_model",
                        help="Directory with converted .safetensors voice presets")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory to save output audio")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default=None, choices=["bfloat16", "float32"],
                        help="Compute dtype, defaults to bfloat16 on cuda and float32 on cpu")
    parser.add_argument("--cfg_scale", type=float, default=1.5, help="CFG scale (default: 1.5)")
    parser.add_argument("--ddpm_steps", type=int, default=5, help="Diffusion inference steps (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--offload_layers_on_gpu", type=int, default=None,
                        help="Enable layer offloading, keeping N of the 20 TTS layers on GPU")
    return parser.parse_args()


def main():
    args = parse_args()
    get_generator(args.seed, force_set=True)

    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if args.device == "cuda" else torch.float32)
    config_path = args.config or os.path.join(args.model_path, "config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = VibeVoiceStreamingConfig.from_dict(config_dict, torch_dtype=dtype)

    offload_config = None
    if args.offload_layers_on_gpu is not None:
        offload_config = OffloadConfig(enabled=True, num_layers_on_gpu=args.offload_layers_on_gpu)

    model_dir = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(config_path)
    processor = VibeVoiceStreamingProcessor.from_pretrained(model_dir)
    model = VibeVoiceStreamingForConditionalInference.from_pretrain(
        args.model_path, config, device=args.device, offload_config=offload_config, dtype=dtype,
    )
    model.eval()
    model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    with open(args.txt_path, "r", encoding="utf-8") as f:
        script = f.read().strip()
    if not script:
        print("Error: No valid script found in the txt file")
        return
    script = script.replace("’", "'").replace("“", '"').replace("”", '"')

    voice_path = find_voice_preset(args.voices_dir, args.speaker_name)
    print(f"Using voice preset for {args.speaker_name}: {voice_path}")
    prefilled_outputs = load_voice_preset(voice_path, device=args.device, dtype=dtype)

    inputs = processor.process_input_with_cached_prompt(text=script, cached_prompt=prefilled_outputs)

    start_time = time.time()
    outputs = model.generate(
        tts_text_ids=inputs["tts_text_ids"],
        tts_lm_input_ids=inputs["tts_lm_input_ids"],
        all_prefilled_outputs=prefilled_outputs,
        cfg_scale=args.cfg_scale,
        verbose=True,
    )
    generation_time = time.time() - start_time

    audio = outputs.speech_outputs[0]
    if audio is None:
        print("No audio output generated")
        return

    sample_rate = 24000
    audio_duration = audio.shape[-1] / sample_rate
    rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

    os.makedirs(args.output_dir, exist_ok=True)
    txt_filename = os.path.splitext(os.path.basename(args.txt_path))[0]
    output_path = os.path.join(args.output_dir, f"{txt_filename}_realtime_generated.wav")
    processor.save_audio(audio, output_path=output_path)

    print("=" * 50)
    print(f"Output file: {output_path}")
    print(f"Speaker: {args.speaker_name}")
    print(f"Text tokens: {inputs['tts_text_ids'].shape[1]}")
    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF (Real Time Factor): {rtf:.2f}x")
    print("=" * 50)


if __name__ == "__main__":
    main()
