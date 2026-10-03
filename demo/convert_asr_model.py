import argparse
import json
import os

import torch

from safetensors.torch import save_model

from config.configuration_vibevoice import DEFAULT_ASR_CONFIG, VibeVoiceASRConfig
from vibevoice.modular.modeling_vibevoice_asr_inference import VibeVoiceASRForConditionalInference


def parse_args():
    parser = argparse.ArgumentParser(description="VibeVoice-ASR Model Convert and Save")
    parser.add_argument("--model_path", type=str, required=True, help="Original HF VibeVoice-ASR model directory")
    parser.add_argument("--type", type=str, default="bfloat16", choices=["bfloat16", "float8_e4m3fn"],
                        help="Data type for the converted model")
    parser.add_argument("--converted_model_name", type=str, default="./vibevoice_asr",
                        help="Output path prefix; the dtype suffix and .safetensors are appended")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device used while converting (cpu needs ~18 GB RAM, no GPU required)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.type == "float8_e4m3fn":
        target_dtype = torch.float8_e4m3fn
        save_model_name = args.converted_model_name + "_float8_e4m3fn.safetensors"
    else:
        target_dtype = torch.bfloat16
        save_model_name = args.converted_model_name + "_bf16.safetensors"

    print(f"The model will be load from {args.model_path}, converted with {args.type} and save to mono file:{save_model_name}")

    config_path = os.path.join(args.model_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"{config_path} not found, using default ASR configuration")
        config_dict = DEFAULT_ASR_CONFIG

    config = VibeVoiceASRConfig.from_dict(config_dict, torch_dtype=torch.bfloat16)

    model = VibeVoiceASRForConditionalInference.from_pretrain(args.model_path, config, device=args.device)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    model.to(dtype=target_dtype)
    print(f"Model converted with dtype {args.type}")

    save_model(model, save_model_name)
    print(f"Model saved to {save_model_name}")


if __name__ == "__main__":
    main()
