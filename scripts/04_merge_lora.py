import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    print("LoRA merge stub.")
    print(f"Base model: {args.base_model}")
    print(f"LoRA dir: {args.lora_dir}")
    print(f"Output dir: {args.output_dir}")
    print("TODO: implement merging logic.")


if __name__ == "__main__":
    main()
