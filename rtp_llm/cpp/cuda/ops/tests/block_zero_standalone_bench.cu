// Standalone benchmark for block_zero kernel.
// Compile: nvcc -O2 -o block_zero_bench block_zero_standalone_bench.cu
// Run:     ./block_zero_bench

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do {                                 \
    cudaError_t e = (call);                                   \
    if (e != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(e));   \
        exit(1);                                              \
    }                                                         \
} while(0)

__global__ void zero_incomplete_kv_cache_blocks_kernel(
    const void* const* __restrict__ layer_base_addrs,
    const int32_t* __restrict__ kv_cache_block_id,
    const int32_t* __restrict__ token_counts,
    const int32_t* __restrict__ layer_to_group,
    size_t batch_size,
    size_t layer_num,
    size_t batch_dim,
    size_t max_blocks_per_batch,
    size_t block_stride_bytes,
    size_t seq_size_per_block)
{
    const size_t batch_idx = blockIdx.x;
    const size_t layer_idx = blockIdx.y;

    if (batch_idx >= batch_size || layer_idx >= layer_num)
        return;

    const int32_t tokens = token_counts[batch_idx];
    if (tokens <= 0)
        return;

    if ((tokens - 1) % static_cast<int32_t>(seq_size_per_block) != 0)
        return;

    const size_t group_idx = layer_to_group ? static_cast<size_t>(layer_to_group[layer_idx]) : 0;

    const size_t last_block_index = static_cast<size_t>(tokens - 1) / seq_size_per_block;
    if (last_block_index >= max_blocks_per_batch)
        return;

    const int32_t block_id =
        kv_cache_block_id[group_idx * batch_dim * max_blocks_per_batch
                          + batch_idx * max_blocks_per_batch
                          + last_block_index];

    if (block_id <= 0)
        return;

    const void* base = layer_base_addrs[layer_idx];
    if (!base)
        return;

    char* dst = static_cast<char*>(const_cast<void*>(base))
                + static_cast<size_t>(block_id) * block_stride_bytes;

    const size_t n_uint4   = block_stride_bytes / sizeof(uint4);
    const size_t remainder = block_stride_bytes % sizeof(uint4);

    uint4*      dst4  = reinterpret_cast<uint4*>(dst);
    const uint4 zero4 = make_uint4(0u, 0u, 0u, 0u);

    for (size_t i = threadIdx.x; i < n_uint4; i += blockDim.x) {
        dst4[i] = zero4;
    }

    if (remainder > 0) {
        const size_t tail_start = n_uint4 * sizeof(uint4);
        for (size_t i = threadIdx.x; i < remainder; i += blockDim.x) {
            dst[tail_start + i] = 0;
        }
    }
}

void invoke(const void* const* layer_base_addrs,
            const int32_t* kv_cache_block_id,
            const int32_t* token_counts,
            size_t batch_size, size_t layer_num, size_t batch_dim,
            size_t max_blocks_per_batch, size_t block_stride_bytes,
            size_t seq_size_per_block, cudaStream_t stream) {
    if (batch_size == 0 || layer_num == 0 || block_stride_bytes == 0)
        return;
    dim3 grid(static_cast<unsigned>(batch_size), static_cast<unsigned>(layer_num));
    dim3 block(256);
    zero_incomplete_kv_cache_blocks_kernel<<<grid, block, 0, stream>>>(
        layer_base_addrs, kv_cache_block_id, token_counts, nullptr,
        batch_size, layer_num, batch_dim, max_blocks_per_batch,
        block_stride_bytes, seq_size_per_block);
}

struct BenchCfg {
    const char* label;
    size_t bs, layers, stride, seq_per_block;
    bool on_boundary;
};

void run_bench(const BenchCfg& cfg, cudaStream_t stream) {
    const int WARMUP = 50;
    const int ITERS  = 500;
    const size_t MAX_BLOCKS = 512;
    const size_t MAX_BPB    = 64;

    // Allocate layer memory
    size_t layer_bytes = MAX_BLOCKS * cfg.stride;
    uint8_t* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, cfg.layers * layer_bytes));
    CHECK_CUDA(cudaMemset(d_buf, 0xAA, cfg.layers * layer_bytes));

    std::vector<void*> h_bases(cfg.layers);
    for (size_t l = 0; l < cfg.layers; ++l)
        h_bases[l] = d_buf + l * layer_bytes;
    void** d_bases = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bases, cfg.layers * sizeof(void*)));
    CHECK_CUDA(cudaMemcpy(d_bases, h_bases.data(), cfg.layers * sizeof(void*), cudaMemcpyHostToDevice));

    // Token counts
    int32_t tok = cfg.on_boundary
                      ? static_cast<int32_t>(cfg.seq_per_block) + 1
                      : static_cast<int32_t>(cfg.seq_per_block) + 2;
    std::vector<int32_t> h_tc(cfg.bs, tok);
    int32_t* d_tc = nullptr;
    CHECK_CUDA(cudaMalloc(&d_tc, cfg.bs * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_tc, h_tc.data(), cfg.bs * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Block IDs
    std::vector<int32_t> h_bids(cfg.bs * MAX_BPB, 0);
    for (size_t b = 0; b < cfg.bs; ++b) {
        size_t needed = (tok - 1) / cfg.seq_per_block + 1;
        for (size_t i = 0; i < std::min(needed, MAX_BPB); ++i) {
            int32_t id = static_cast<int32_t>(b * 2 + i + 1);
            if (id < static_cast<int32_t>(MAX_BLOCKS))
                h_bids[b * MAX_BPB + i] = id;
        }
    }
    int32_t* d_bids = nullptr;
    CHECK_CUDA(cudaMalloc(&d_bids, h_bids.size() * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_bids, h_bids.data(), h_bids.size() * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP; ++i) {
        invoke(reinterpret_cast<const void* const*>(d_bases),
               d_bids, d_tc, cfg.bs, cfg.layers, cfg.bs,
               MAX_BPB, cfg.stride, cfg.seq_per_block, stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Timed
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERS; ++i) {
        invoke(reinterpret_cast<const void* const*>(d_bases),
               d_bids, d_tc, cfg.bs, cfg.layers, cfg.bs,
               MAX_BPB, cfg.stride, cfg.seq_per_block, stream);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_us = total_ms * 1000.0f / ITERS;

    double data_mb = 0;
    if (cfg.on_boundary) {
        data_mb = static_cast<double>(cfg.bs) * cfg.layers * cfg.stride / (1024.0 * 1024.0);
    }

    printf("  %-38s %8.2f us    %8.2f MB\n", cfg.label, avg_us, data_mb);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFree(d_bases));
    CHECK_CUDA(cudaFree(d_tc));
    CHECK_CUDA(cudaFree(d_bids));
}

int main() {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("\nGPU: %s\n\n", prop.name);

    // stride = head_dim * kv_heads * 2(k+v) * seq_per_block * sizeof(fp16)
    // 8-head GQA, spb=1: 128*8*2*1*2 = 4096
    // 8-head GQA, spb=16: 128*8*2*16*2 = 65536
    // 4-head GQA, spb=1: 128*4*2*1*2 = 2048

    const BenchCfg configs[] = {
        // === Worst case: all batches on new-block boundary ===
        {"bs=1   L=32  stride=4K   boundary", 1,   32, 4096,  1, true},
        {"bs=1   L=80  stride=4K   boundary", 1,   80, 4096,  1, true},
        {"bs=8   L=32  stride=4K   boundary", 8,   32, 4096,  1, true},
        {"bs=8   L=80  stride=4K   boundary", 8,   80, 4096,  1, true},
        {"bs=32  L=32  stride=4K   boundary", 32,  32, 4096,  1, true},
        {"bs=32  L=80  stride=4K   boundary", 32,  80, 4096,  1, true},
        {"bs=128 L=32  stride=4K   boundary", 128, 32, 4096,  1, true},
        {"bs=128 L=80  stride=4K   boundary", 128, 80, 4096,  1, true},
        {"bs=256 L=32  stride=4K   boundary", 256, 32, 4096,  1, true},
        {"bs=256 L=80  stride=4K   boundary", 256, 80, 4096,  1, true},

        // Larger block stride (seq_per_block=16)
        {"bs=32  L=32  stride=64K  boundary", 32,  32, 65536, 16, true},
        {"bs=128 L=80  stride=64K  boundary", 128, 80, 65536, 16, true},
        {"bs=256 L=80  stride=64K  boundary", 256, 80, 65536, 16, true},

        // Smaller stride (4-head GQA)
        {"bs=128 L=32  stride=2K   boundary", 128, 32, 2048,  1, true},

        // === No-op: none on boundary ===
        {"bs=1   L=32  stride=4K   no-op",    1,   32, 4096,  4, false},
        {"bs=32  L=32  stride=4K   no-op",    32,  32, 4096,  4, false},
        {"bs=128 L=80  stride=4K   no-op",    128, 80, 4096,  4, false},
        {"bs=256 L=80  stride=4K   no-op",    256, 80, 4096,  4, false},
        {"bs=256 L=80  stride=64K  no-op",    256, 80, 65536, 16, false},
    };

    int n = sizeof(configs) / sizeof(configs[0]);

    printf("==========================================================================\n");
    printf("  BlockZero Kernel Benchmark  (warmup=%d, iters=%d)\n", 50, 500);
    printf("==========================================================================\n");
    printf("  %-38s %8s      %8s\n", "Config", "Avg", "Data/call");
    printf("--------------------------------------------------------------------------\n");

    for (int i = 0; i < n; ++i) {
        run_bench(configs[i], stream);
    }

    printf("==========================================================================\n");
    printf("\nNote: 'boundary' = worst case (all batches zero one block per layer)\n");
    printf("      'no-op'    = best case (kernel early-returns, no memset)\n");
    printf("      Data/call  = total bytes zeroed per invocation\n\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
