import torch
import triton

from rtp_llm.models_py.modules import QKRMSNorm, FusedQKRMSNorm


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],
        x_vals=[
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            32768,
            65536,
            131072
        ],
        line_arg="provider",
        line_vals=["qkrmsnorm", "fusedqkrmsnorm"],
        line_names=["QKRMSNorm", "FusedQKRMSNorm"],
        styles=[("red", "-"), ("blue", "-"), ("green", "-")],
        xlabel="Num Tokens",
        ylabel="Latency (ms)",
        plot_name="qkrmsnorm-latency",
        args={
            "head_num": 64,
            "kv_head_num": 8,
            "size_per_head": 128,
            "dtype": torch.bfloat16,
            "device": "cuda",
        },
    )
)
def benchmark(
    provider,
    num_tokens: int,
    head_num: int,
    kv_head_num: int,
    size_per_head: int,
    dtype: torch.dtype,
    device: torch.device
):
    print(f"provider: {provider}, num_tokens: {num_tokens}, head_num: {head_num}, kv_head_num: {kv_head_num}, size_per_head: {size_per_head}, dtype: {dtype}, device: {device}")

    q_weight = torch.randn(size_per_head, dtype=dtype, device=device)
    k_weight = torch.randn(size_per_head, dtype=dtype, device=device)

    qkrmsnorm_forward = None
    if provider == "qkrmsnorm":
        qkrmsnorm = QKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        qkrmsnorm_forward = qkrmsnorm.forward
    elif provider == "fusedqkrmsnorm":
        qkrmsnorm = FusedQKRMSNorm(q_weight, k_weight, head_num, kv_head_num, size_per_head)
        qkrmsnorm_forward = qkrmsnorm.forward

    input = torch.randn(num_tokens, (head_num + 2 * kv_head_num) * size_per_head, dtype=dtype, device=device)

    return triton.testing.do_bench(lambda: qkrmsnorm_forward(input))


if __name__ == "__main__":
    benchmark.run(print_data=True, show_plots=True, save_path="qkrmsnorm")
