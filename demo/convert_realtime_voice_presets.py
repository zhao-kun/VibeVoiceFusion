import argparse
import glob
import os

from util.streaming_voice_preset import convert_pt_voice_preset


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VibeVoice Realtime .pt voice presets to safetensors")
    parser.add_argument("--input", type=str, required=True,
                        help="A .pt preset file, or a directory searched recursively for .pt presets")
    parser.add_argument("--output_dir", type=str, default="demo/voices/streaming_model",
                        help="Directory to write <name>.safetensors presets")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted presets")
    return parser.parse_args()


def main():
    args = parse_args()
    if os.path.isdir(args.input):
        pt_files = sorted(glob.glob(os.path.join(args.input, "**", "*.pt"), recursive=True))
    else:
        pt_files = [args.input]

    if not pt_files:
        print(f"No .pt presets found in {args.input}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print("Note: .pt presets are unpickled; only convert files from a trusted source.")
    for pt_file in pt_files:
        name = os.path.splitext(os.path.basename(pt_file))[0]
        output_path = os.path.join(args.output_dir, f"{name}.safetensors")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skip {name}: {output_path} exists")
            continue
        convert_pt_voice_preset(pt_file, output_path)
        print(f"Converted {pt_file} -> {output_path}")


if __name__ == "__main__":
    main()
