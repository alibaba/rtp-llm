# Qwen3 MoE

Qwen3-235B-A22B is smaller in model size compared to DeepSeek-V3 but supports seamless Thinking Mode switching. For 4K Input/2K Output scenarios, similar optimization strategies can be adopted, adjusting parallel modes in combination with specific model parameter configurations.

From the KV Cache usage perspective, Qwen3-235B-A22B's per-token KV Cache overhead is 94×4×128×2=96KB, while DeepSeek-V3 is 61×1536=93KB, which are close.

From the Attention computation latency perspective, Qwen3-235B-A22B uses 64-head GQA, while DeepSeek-V3 uses 128-head MLA, with computation latency being approximately 50% of the latter. Considering the impact of memory access latency, actual latency will be slightly higher.

From the Dispatch/Combine communication perspective, Qwen3-235B-A22B is about 40% of DeepSeek-V3.

From the MoE GEMM computation latency perspective, due to Qwen3-235B-A22B's parameter scale being 40%-50%, the computation latency is about 50%.

In summary, in large-scale cluster deployment, comprehensively evaluating from the two dimensions of KV Cache capacity limitations and MoE computational efficiency, Qwen3-235B-A22B can adopt similar deployment modes. Compared to DeepSeek-V3, Qwen3-235B-A22B can support longer sequence lengths with better performance in terms of latency and throughput. For compute-constrained cards like H20, EP can be reduced and TP introduced to reduce network latency while achieving good computational utilization.