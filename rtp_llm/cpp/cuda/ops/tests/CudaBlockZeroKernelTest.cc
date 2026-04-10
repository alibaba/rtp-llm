#include "rtp_llm/cpp/cuda/ops/tests/CudaTestUtils.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/kernels/block_zero_kernels.h"
#include <cuda_runtime.h>
#include <vector>

using namespace rtp_llm;

class CudaBlockZeroKernelTest: public DeviceTestBase {
protected:
    void SetUp() override {
        DeviceTestBase::SetUp();
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        DeviceTestBase::TearDown();
    }
    cudaStream_t stream_ = nullptr;

    struct GpuMem {
        uint8_t* buf   = nullptr;
        void**   bases = nullptr;
        ~GpuMem() {
            cudaFree(buf);
            cudaFree(bases);
        }
    };

    GpuMem setupMemory(size_t layers, size_t blocks, size_t stride, uint8_t fill) {
        GpuMem m;
        size_t layer_bytes = blocks * stride;
        cudaMalloc(&m.buf, layers * layer_bytes);
        cudaMemset(m.buf, fill, layers * layer_bytes);

        std::vector<void*> h_bases(layers);
        for (size_t l = 0; l < layers; ++l)
            h_bases[l] = m.buf + l * layer_bytes;
        cudaMalloc(&m.bases, layers * sizeof(void*));
        cudaMemcpy(m.bases, h_bases.data(), layers * sizeof(void*), cudaMemcpyHostToDevice);
        return m;
    }

    template <typename T>
    T* toDevice(const std::vector<T>& h) {
        T* d = nullptr;
        cudaMalloc(&d, h.size() * sizeof(T));
        cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
        return d;
    }

    std::vector<uint8_t> download(uint8_t* d_ptr, size_t bytes) {
        std::vector<uint8_t> h(bytes);
        cudaMemcpy(h.data(), d_ptr, bytes, cudaMemcpyDeviceToHost);
        return h;
    }

    bool isAllValue(const uint8_t* data, size_t len, uint8_t val) {
        for (size_t i = 0; i < len; ++i)
            if (data[i] != val)
                return false;
        return true;
    }
};

// Zeros block only at new-block boundary: (tokens-1) % seq_size_per_block == 0.
TEST_F(CudaBlockZeroKernelTest, ZerosOnlyNewBlockBoundary) {
    constexpr size_t kLayers         = 2;
    constexpr size_t kBlocks         = 8;
    constexpr size_t kStride         = 256;
    constexpr size_t kSeqPerBlock    = 4;
    constexpr size_t kBatchDim       = 2;
    constexpr size_t kMaxBlocksBatch = 4;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xCC);

    // batch 0: tokens=9, (9-1)%4==0 → boundary → zero block at idx 2
    //   block_ids: [1,2,5,0] → idx 2 → block_id 5 → ZEROED
    // batch 1: tokens=1, (1-1)%4==0 → boundary → zero block at idx 0
    //   block_ids: [3,0,0,0] → idx 0 → block_id 3 → ZEROED
    std::vector<int32_t> token_counts = {9, 1};
    std::vector<int32_t> block_ids    = {1,2,5,0, 3,0,0,0};

    auto* d_tc   = toDevice(token_counts);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, nullptr,
        2, kLayers, kBatchDim, kMaxBlocksBatch, kStride, kSeqPerBlock, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kLayers * kBlocks * kStride);
    for (size_t l = 0; l < kLayers; ++l) {
        const uint8_t* layer = result.data() + l * kBlocks * kStride;
        EXPECT_TRUE(isAllValue(layer + 5 * kStride, kStride, 0x00)) << "l=" << l << " block 5 zeroed";
        EXPECT_TRUE(isAllValue(layer + 3 * kStride, kStride, 0x00)) << "l=" << l << " block 3 zeroed";
        EXPECT_TRUE(isAllValue(layer + 1 * kStride, kStride, 0xCC)) << "l=" << l << " block 1 untouched";
        EXPECT_TRUE(isAllValue(layer + 2 * kStride, kStride, 0xCC)) << "l=" << l << " block 2 untouched";
    }

    cudaFree(d_tc);
    cudaFree(d_bids);
}

// Mid-block tokens are skipped by the kernel's modulo guard — no host-side filtering needed.
TEST_F(CudaBlockZeroKernelTest, SkipsMidBlockTokens) {
    constexpr size_t kLayers         = 1;
    constexpr size_t kBlocks         = 8;
    constexpr size_t kStride         = 256;
    constexpr size_t kSeqPerBlock    = 4;
    constexpr size_t kBatchDim       = 2;
    constexpr size_t kMaxBlocksBatch = 4;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xCC);

    // batch 0: tokens=10, (10-1)%4==1 → NOT boundary → skip
    // batch 1: tokens=3,  (3-1)%4==2  → NOT boundary → skip
    std::vector<int32_t> token_counts = {10, 3};
    std::vector<int32_t> block_ids    = {1,2,5,0, 3,0,0,0};

    auto* d_tc   = toDevice(token_counts);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, nullptr,
        2, kLayers, kBatchDim, kMaxBlocksBatch, kStride, kSeqPerBlock, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kBlocks * kStride);
    EXPECT_TRUE(isAllValue(result.data(), result.size(), 0xCC)) << "everything untouched";

    cudaFree(d_tc);
    cudaFree(d_bids);
}

// Multi-group: layer_to_group routes layers to different block_id groups.
TEST_F(CudaBlockZeroKernelTest, MultiGroupLayerMapping) {
    constexpr size_t kLayers         = 2;
    constexpr size_t kBlocks         = 8;
    constexpr size_t kStride         = 128;
    constexpr size_t kSeqPerBlock    = 4;
    constexpr size_t kBatchDim       = 1;
    constexpr size_t kMaxBlocksBatch = 4;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xBB);

    // tokens=5, (5-1)%4==0 → boundary → zero block at idx 1
    // layer 0 -> group 0: block_ids [1,3,0,0] -> idx 1 -> block_id 3
    // layer 1 -> group 1: block_ids [2,6,0,0] -> idx 1 -> block_id 6
    std::vector<int32_t> token_counts    = {5};
    std::vector<int32_t> layer_to_group  = {0, 1};
    std::vector<int32_t> block_ids       = {1,3,0,0, 2,6,0,0};

    auto* d_tc  = toDevice(token_counts);
    auto* d_ltg = toDevice(layer_to_group);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, d_ltg,
        1, kLayers, kBatchDim, kMaxBlocksBatch, kStride, kSeqPerBlock, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kLayers * kBlocks * kStride);
    EXPECT_TRUE(isAllValue(result.data() + 0 * kBlocks * kStride + 3 * kStride, kStride, 0x00))
        << "layer 0 block 3 zeroed (group 0)";
    EXPECT_TRUE(isAllValue(result.data() + 0 * kBlocks * kStride + 1 * kStride, kStride, 0xBB))
        << "layer 0 block 1 untouched";
    EXPECT_TRUE(isAllValue(result.data() + 1 * kBlocks * kStride + 6 * kStride, kStride, 0x00))
        << "layer 1 block 6 zeroed (group 1)";
    EXPECT_TRUE(isAllValue(result.data() + 1 * kBlocks * kStride + 2 * kStride, kStride, 0xBB))
        << "layer 1 block 2 untouched";

    cudaFree(d_tc);
    cudaFree(d_ltg);
    cudaFree(d_bids);
}

// Skips invalid block IDs (<=0) and zero token counts.
TEST_F(CudaBlockZeroKernelTest, SkipsInvalidBlocksAndZeroTokens) {
    constexpr size_t kLayers         = 1;
    constexpr size_t kBlocks         = 4;
    constexpr size_t kStride         = 64;
    constexpr size_t kBatchDim       = 2;
    constexpr size_t kMaxBlocksBatch = 2;

    auto mem = setupMemory(kLayers, kBlocks, kStride, 0xFF);

    // batch 0: token_counts=0 -> skip (tokens <= 0)
    // batch 1: token_counts=1 -> (1-1)%4==0 boundary, but block_id=-1 -> skip (invalid)
    std::vector<int32_t> token_counts = {0, 1};
    std::vector<int32_t> block_ids    = {0,0, -1,0};

    auto* d_tc   = toDevice(token_counts);
    auto* d_bids = toDevice(block_ids);

    invokeZeroIncompleteKvCacheBlocks(
        reinterpret_cast<const void* const*>(mem.bases),
        d_bids, d_tc, nullptr,
        2, kLayers, kBatchDim, kMaxBlocksBatch, kStride, 4, stream_);
    cudaStreamSynchronize(stream_);

    auto result = download(mem.buf, kBlocks * kStride);
    EXPECT_TRUE(isAllValue(result.data(), result.size(), 0xFF)) << "everything untouched";

    cudaFree(d_tc);
    cudaFree(d_bids);
}

// Empty inputs: no crash, no-op.
TEST_F(CudaBlockZeroKernelTest, EmptyInputNoOp) {
    invokeZeroIncompleteKvCacheBlocks(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 256, 4, stream_);
    invokeZeroIncompleteKvCacheBlocks(nullptr, nullptr, nullptr, nullptr, 0, 4, 0, 0, 256, 4, stream_);
    invokeZeroIncompleteKvCacheBlocks(nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 4, stream_);
    cudaStreamSynchronize(stream_);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// ─── Performance benchmarks ────────────────────────────────────────────────

struct BenchConfig {
    const char* label;
    size_t batch_size;
    size_t layer_num;
    size_t block_stride_bytes;
    size_t seq_size_per_block;
    bool   all_on_boundary;
};

static const char* variantName(BlockZeroVariant v) {
    switch (v) {
        case BlockZeroVariant::kLayerFused:      return "Fused";
        case BlockZeroVariant::kLayerFusedCG:    return "Fused+CG";
        case BlockZeroVariant::kLayerParallel:   return "Parallel";
        case BlockZeroVariant::kLayerParallelCG: return "Para+CG";
        case BlockZeroVariant::kLayerPerBlock:   return "PerBlock";
    }
    return "?";
}

TEST_F(CudaBlockZeroKernelTest, PerfBenchmarkVariants) {
    constexpr int kWarmup = 200;
    constexpr int kIters  = 1000;
    constexpr size_t kMaxBlocks = 512;
    constexpr size_t kMaxBlocksPerBatch = 64;

    const BenchConfig configs[] = {
        // --- Decode: MHA GQA BF16 (Qwen2.5-72B / Llama-70B) L=80 stride=32K ---
        {"decode GQA-BF16  bs=1   L=80",   1,   80, 32768,  8, true},
        {"decode GQA-BF16  bs=32  L=80",   32,  80, 32768,  8, true},
        {"decode GQA-BF16  bs=128 L=80",   128, 80, 32768,  8, true},
        {"decode GQA-BF16  bs=256 L=80",   256, 80, 32768,  8, true},

        // --- Decode: MHA GQA FP8 (Qwen2.5-72B) L=80 stride=16K ---
        {"decode GQA-FP8   bs=32  L=80",   32,  80, 16384,  8, true},
        {"decode GQA-FP8   bs=128 L=80",   128, 80, 16384,  8, true},
        {"decode GQA-FP8   bs=256 L=80",   256, 80, 16384,  8, true},

        // --- Decode: MLA BF16 (DeepSeek-V3) L=61 stride=9216 ---
        {"decode MLA-BF16  bs=1   L=61",   1,   61, 9216,   8, true},
        {"decode MLA-BF16  bs=32  L=61",   32,  61, 9216,   8, true},
        {"decode MLA-BF16  bs=128 L=61",   128, 61, 9216,   8, true},
        {"decode MLA-BF16  bs=256 L=61",   256, 61, 9216,   8, true},

        // --- Decode: MLA FP8 (DeepSeek-V3) L=61 stride=5248 ---
        {"decode MLA-FP8   bs=128 L=61",   128, 61, 5248,   8, true},
        {"decode MLA-FP8   bs=256 L=61",   256, 61, 5248,   8, true},

        // --- Decode: MQA BF16 (small) L=32 stride=4K ---
        {"decode MQA-BF16  bs=128 L=32",   128, 32, 4096,   8, true},
        {"decode MQA-BF16  bs=256 L=32",   256, 32, 4096,   8, true},

        // --- Decode: Llama-8B GQA BF16 L=32 stride=32K ---
        {"decode 8B-BF16   bs=32  L=32",   32,  32, 32768,  8, true},
        {"decode 8B-BF16   bs=128 L=32",   128, 32, 32768,  8, true},

        // --- Prefill: small batch, same strides ---
        {"prefill GQA-BF16 bs=1   L=80",   1,   80, 32768,  8, true},
        {"prefill GQA-BF16 bs=4   L=80",   4,   80, 32768,  8, true},
        {"prefill MLA-BF16 bs=1   L=61",   1,   61, 9216,   8, true},
        {"prefill MLA-BF16 bs=4   L=61",   4,   61, 9216,   8, true},

        // --- No-op (not at boundary) per stride ---
        {"no-op  GQA-BF16  bs=128 L=80",   128, 80, 32768,  8, false},
        {"no-op  GQA-BF16  bs=256 L=80",   256, 80, 32768,  8, false},
        {"no-op  MLA-BF16  bs=128 L=61",   128, 61, 9216,   8, false},
        {"no-op  MLA-BF16  bs=256 L=61",   256, 61, 9216,   8, false},
        {"no-op  MQA-BF16  bs=256 L=32",   256, 32, 4096,   8, false},
    };

    const BlockZeroVariant variants[] = {
        BlockZeroVariant::kLayerFused,
        BlockZeroVariant::kLayerParallel,
        BlockZeroVariant::kLayerPerBlock,
    };
    constexpr int kNumVariants = 3;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("\n%-38s  %10s  %10s  %10s  %10s\n",
           "Config", "Data(MB)", variantName(variants[0]),
           variantName(variants[1]), variantName(variants[2]));
    printf("%s\n", std::string(90, '-').c_str());

    for (const auto& cfg : configs) {
        auto mem = setupMemory(cfg.layer_num, kMaxBlocks, cfg.block_stride_bytes, 0xAA);

        int32_t tok = cfg.all_on_boundary
                          ? static_cast<int32_t>(cfg.seq_size_per_block) + 1
                          : static_cast<int32_t>(cfg.seq_size_per_block) + 2;
        std::vector<int32_t> token_counts(cfg.batch_size, tok);

        std::vector<int32_t> block_ids(cfg.batch_size * kMaxBlocksPerBatch, 0);
        for (size_t b = 0; b < cfg.batch_size; ++b) {
            size_t needed = (tok - 1) / cfg.seq_size_per_block + 1;
            for (size_t i = 0; i < std::min(needed, kMaxBlocksPerBatch); ++i) {
                int32_t id = static_cast<int32_t>(b * kMaxBlocksPerBatch + i + 1);
                if (id < static_cast<int32_t>(kMaxBlocks))
                    block_ids[b * kMaxBlocksPerBatch + i] = id;
            }
        }

        auto* d_tc   = toDevice(token_counts);
        auto* d_bids = toDevice(block_ids);

        double data_mb = cfg.all_on_boundary
            ? static_cast<double>(cfg.batch_size) * cfg.layer_num
              * cfg.block_stride_bytes / (1024.0 * 1024.0)
            : 0.0;

        float results_us[kNumVariants] = {};

        for (int vi = 0; vi < kNumVariants; ++vi) {
            for (int i = 0; i < kWarmup; ++i) {
                invokeZeroIncompleteKvCacheBlocksVariant(
                    reinterpret_cast<const void* const*>(mem.bases),
                    d_bids, d_tc, nullptr,
                    cfg.batch_size, cfg.layer_num, cfg.batch_size,
                    kMaxBlocksPerBatch, cfg.block_stride_bytes,
                    cfg.seq_size_per_block, stream_, variants[vi]);
            }
            cudaStreamSynchronize(stream_);

            cudaEventRecord(start, stream_);
            for (int i = 0; i < kIters; ++i) {
                invokeZeroIncompleteKvCacheBlocksVariant(
                    reinterpret_cast<const void* const*>(mem.bases),
                    d_bids, d_tc, nullptr,
                    cfg.batch_size, cfg.layer_num, cfg.batch_size,
                    kMaxBlocksPerBatch, cfg.block_stride_bytes,
                    cfg.seq_size_per_block, stream_, variants[vi]);
            }
            cudaEventRecord(stop, stream_);
            cudaEventSynchronize(stop);

            float total_ms = 0;
            cudaEventElapsedTime(&total_ms, start, stop);
            results_us[vi] = total_ms * 1000.0f / kIters;
        }

        printf("%-38s  %8.1f MB  %8.2f us  %8.2f us  %8.2f us\n",
               cfg.label, data_mb,
               results_us[0], results_us[1], results_us[2]);

        cudaFree(d_tc);
        cudaFree(d_bids);
    }

    printf("\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}
