import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora.yaml")
    parser.add_argument("--train-data", required=False)
    args = parser.parse_args()

    print("LoRA training stub.")
    print(f"Config: {args.config}")
    if args.train_data:
        print(f"Train data: {args.train_data}")
    print("TODO: implement with your preferred trainer.")


if __name__ == "__main__":
    main()
