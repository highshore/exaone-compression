import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
    )
    params = SamplingParams(temperature=0.0, max_tokens=32)
    outputs = llm.generate(["hello"], params)
    print(outputs[0].outputs[0].text)
    print("OK: vLLM load + generate succeeded")


if __name__ == "__main__":
    main()
