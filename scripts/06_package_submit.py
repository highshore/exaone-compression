import argparse
import os
import shutil
import tempfile
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", default="submit.zip")
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    output_zip = os.path.abspath(args.output)

    if not os.path.isdir(model_dir):
        raise SystemExit(f"model-dir not found: {model_dir}")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_target = os.path.join(tmpdir, "model")
        shutil.copytree(model_dir, model_target)

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(model_target):
                for name in files:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, tmpdir)
                    zf.write(full_path, rel_path)

    print(f"Created {output_zip}")


if __name__ == "__main__":
    main()
