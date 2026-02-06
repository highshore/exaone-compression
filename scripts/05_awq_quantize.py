import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/awq.json")
    args = parser.parse_args()

    print("AWQ quantization stub.")
    print(f"Model dir: {args.model_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Config: {args.config}")
    print("TODO: implement AWQ with autoawq.")


if __name__ == "__main__":
    main()
