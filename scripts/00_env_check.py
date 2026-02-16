import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch
import vllm
import transformers


EXPECTED = {
    "python_prefix": "3.11.",
    "torch": "2.9.0+cu128",
    "vllm": "0.14.1",
    "transformers": "4.57.3",
}


def main() -> None:
    print(f"python: {platform.python_version()}")
    print(f"torch: {torch.__version__}")
    print(f"vllm: {vllm.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    try:
        autoawq_version = version("autoawq")
    except PackageNotFoundError:
        autoawq_version = None
    print(f"autoawq: {autoawq_version or 'not-installed'}")

    issues = []
    if not platform.python_version().startswith(EXPECTED["python_prefix"]):
        issues.append("python version mismatch")
    if torch.__version__ != EXPECTED["torch"]:
        issues.append("torch version mismatch")
    if vllm.__version__ != EXPECTED["vllm"]:
        issues.append("vllm version mismatch")
    if transformers.__version__ != EXPECTED["transformers"]:
        issues.append("transformers version mismatch")

    if issues:
        print("WARN: environment mismatch detected")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)
    print("OK: environment matches expected versions")


if __name__ == "__main__":
    main()
