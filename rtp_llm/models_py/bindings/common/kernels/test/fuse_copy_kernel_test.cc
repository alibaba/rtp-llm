#include <gtest/gtest.h>
#include <cstring>
#include <vector>

#if USING_CUDA
#include <cuda_runtime.h>
#elif USING_ROCM
#include <hip/hip_runtime.h>
#include "rtp_llm/cpp/rocm/cuda_shims.h"
#endif

#include "rtp_llm/models_py/bindings/common/kernels/fuse_copy_kernel.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

#define CUDA_CHECK(expr)                                                                                               \
    do {                                                                                                               \
        cudaError_t _e = (expr);                                                                                       \
        ASSERT_EQ(_e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(_e);                                        \
    } while (0)

// Allocate device memory and copy host data to it.
template<typename T>
T* deviceAlloc(const std::vector<T>& host_data) {
    T* d_ptr = nullptr;
    EXPECT_EQ(cudaMalloc(&d_ptr, host_data.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_ptr, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d_ptr;
}

// Allocate zero-initialised device memory.
template<typename T>
T* deviceAllocZero(size_t n) {
    T* d_ptr = nullptr;
    EXPECT_EQ(cudaMalloc(&d_ptr, n * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemset(d_ptr, 0, n * sizeof(T)), cudaSuccess);
    return d_ptr;
}

// Copy device data back to a host vector.
template<typename T>
std::vector<T> deviceToHost(const T* d_ptr, size_t n) {
    std::vector<T> host(n);
    EXPECT_EQ(cudaMemcpy(host.data(), d_ptr, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
}

}  // namespace

// ---------------------------------------------------------------------------
// FusedCopy tests (invokeFusedCopy)
// ---------------------------------------------------------------------------

class FusedCopyTest: public ::testing::Test {
protected:
    cudaStream_t stream_{};

    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
    }
};

// num_copies == 0 should be a no-op; ensure no crash.
TEST_F(FusedCopyTest, ZeroCopies) {
    rtp_llm::FusedD2DCopyParams params;
    rtp_llm::invokeFusedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// Single aligned copy (16-byte vectorised fast path).
TEST_F(FusedCopyTest, SingleAlignedCopy) {
    constexpr size_t     N = 1024;  // 1024 bytes, 16-byte aligned
    std::vector<uint8_t> host_src(N);
    for (size_t i = 0; i < N; ++i)
        host_src[i] = static_cast<uint8_t>(i & 0xFF);

    uint8_t* d_src = deviceAlloc(host_src);
    uint8_t* d_dst = deviceAllocZero<uint8_t>(N);

    rtp_llm::FusedD2DCopyParams params;
    params.add(d_src, d_dst, N);

    rtp_llm::invokeFusedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto result = deviceToHost(d_dst, N);
    for (size_t i = 0; i < N; ++i)
        ASSERT_EQ(result[i], host_src[i]) << "mismatch at byte " << i;

    cudaFree(d_src);
    cudaFree(d_dst);
}

// Unaligned copy: shift dst by 1 byte so the slow (byte-by-byte) path triggers.
TEST_F(FusedCopyTest, UnalignedDstCopy) {
    constexpr size_t N = 128;

    std::vector<uint8_t> host_src(N);
    for (size_t i = 0; i < N; ++i)
        host_src[i] = static_cast<uint8_t>((i * 7 + 3) & 0xFF);

    // Allocate a buffer that is 1 byte larger, then offset dst by 1 to break alignment.
    uint8_t* d_src      = deviceAlloc(host_src);
    uint8_t* d_dst_base = deviceAllocZero<uint8_t>(N + 1);
    uint8_t* d_dst      = d_dst_base + 1;  // unaligned

    rtp_llm::FusedD2DCopyParams params;
    params.add(d_src, d_dst, N);

    rtp_llm::invokeFusedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto result = deviceToHost(d_dst, N);
    for (size_t i = 0; i < N; ++i)
        ASSERT_EQ(result[i], host_src[i]) << "mismatch at byte " << i;

    cudaFree(d_src);
    cudaFree(d_dst_base);
}

// Copy where size is not a multiple of 16 — exercises the remainder loop.
TEST_F(FusedCopyTest, NonMultipleOf16Size) {
    constexpr size_t N = 37;  // deliberately not a multiple of 16

    std::vector<uint8_t> host_src(N);
    for (size_t i = 0; i < N; ++i)
        host_src[i] = static_cast<uint8_t>(i);

    uint8_t* d_src = deviceAlloc(host_src);
    uint8_t* d_dst = deviceAllocZero<uint8_t>(N);

    rtp_llm::FusedD2DCopyParams params;
    params.add(d_src, d_dst, N);

    rtp_llm::invokeFusedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto result = deviceToHost(d_dst, N);
    for (size_t i = 0; i < N; ++i)
        ASSERT_EQ(result[i], host_src[i]) << "mismatch at byte " << i;

    cudaFree(d_src);
    cudaFree(d_dst);
}

// Multiple copies batched into one kernel launch.
TEST_F(FusedCopyTest, MultipleCopies) {
    const std::vector<size_t> sizes = {64, 128, 256, 512};

    std::vector<std::vector<uint8_t>> host_srcs(sizes.size());
    std::vector<uint8_t*>             d_srcs(sizes.size());
    std::vector<uint8_t*>             d_dsts(sizes.size());

    for (size_t c = 0; c < sizes.size(); ++c) {
        host_srcs[c].resize(sizes[c]);
        for (size_t i = 0; i < sizes[c]; ++i)
            host_srcs[c][i] = static_cast<uint8_t>((c * 13 + i) & 0xFF);
        d_srcs[c] = deviceAlloc(host_srcs[c]);
        d_dsts[c] = deviceAllocZero<uint8_t>(sizes[c]);
    }

    rtp_llm::FusedD2DCopyParams params;
    for (size_t c = 0; c < sizes.size(); ++c)
        params.add(d_srcs[c], d_dsts[c], sizes[c]);

    rtp_llm::invokeFusedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    for (size_t c = 0; c < sizes.size(); ++c) {
        auto result = deviceToHost(d_dsts[c], sizes[c]);
        for (size_t i = 0; i < sizes[c]; ++i)
            ASSERT_EQ(result[i], host_srcs[c][i]) << "copy " << c << " mismatch at byte " << i;
    }

    for (size_t c = 0; c < sizes.size(); ++c) {
        cudaFree(d_srcs[c]);
        cudaFree(d_dsts[c]);
    }
}

// Fill MAX_FUSED_D2D_COPIES copies to stress the capacity limit.
TEST_F(FusedCopyTest, MaxFusedCopies) {
    constexpr size_t N = 256;

    std::vector<std::vector<uint8_t>> host_srcs(rtp_llm::MAX_FUSED_D2D_COPIES);
    std::vector<uint8_t*>             d_srcs(rtp_llm::MAX_FUSED_D2D_COPIES);
    std::vector<uint8_t*>             d_dsts(rtp_llm::MAX_FUSED_D2D_COPIES);

    for (int c = 0; c < rtp_llm::MAX_FUSED_D2D_COPIES; ++c) {
        host_srcs[c].resize(N);
        for (size_t i = 0; i < N; ++i)
            host_srcs[c][i] = static_cast<uint8_t>((c * 17 + i) & 0xFF);
        d_srcs[c] = deviceAlloc(host_srcs[c]);
        d_dsts[c] = deviceAllocZero<uint8_t>(N);
    }

    rtp_llm::FusedD2DCopyParams params;
    for (int c = 0; c < rtp_llm::MAX_FUSED_D2D_COPIES; ++c)
        params.add(d_srcs[c], d_dsts[c], N);

    rtp_llm::invokeFusedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    for (int c = 0; c < rtp_llm::MAX_FUSED_D2D_COPIES; ++c) {
        auto result = deviceToHost(d_dsts[c], N);
        for (size_t i = 0; i < N; ++i)
            ASSERT_EQ(result[i], host_srcs[c][i]) << "copy " << c << " mismatch at byte " << i;
    }

    for (int c = 0; c < rtp_llm::MAX_FUSED_D2D_COPIES; ++c) {
        cudaFree(d_srcs[c]);
        cudaFree(d_dsts[c]);
    }
}

// ---------------------------------------------------------------------------
// FusedStridedCopy tests (invokeFusedStridedCopy)
// ---------------------------------------------------------------------------

class FusedStridedCopyTest: public ::testing::Test {
protected:
    cudaStream_t stream_{};

    void SetUp() override {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    void TearDown() override {
        cudaStreamDestroy(stream_);
    }
};

// num_copies == 0 should be a no-op.
TEST_F(FusedStridedCopyTest, ZeroCopies) {
    rtp_llm::FusedStridedCopyParams params;
    rtp_llm::invokeFusedStridedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

// Basic strided copy: src_stride > row_bytes (skip padding bytes in source).
// Layout: src has rows of src_stride bytes, only row_bytes are valid data.
//         dst is compact (dst_stride == row_bytes).
TEST_F(FusedStridedCopyTest, SingleStridedCopy) {
    constexpr size_t NROWS      = 8;
    constexpr size_t ROW_BYTES  = 32;
    constexpr size_t SRC_STRIDE = 64;         // each source row is 64 bytes wide
    constexpr size_t DST_STRIDE = ROW_BYTES;  // compact destination

    std::vector<uint8_t> host_src(NROWS * SRC_STRIDE, 0xAB);
    // Fill only the valid data region.
    for (size_t r = 0; r < NROWS; ++r)
        for (size_t b = 0; b < ROW_BYTES; ++b)
            host_src[r * SRC_STRIDE + b] = static_cast<uint8_t>((r * ROW_BYTES + b) & 0xFF);

    uint8_t* d_src = deviceAlloc(host_src);
    uint8_t* d_dst = deviceAllocZero<uint8_t>(NROWS * DST_STRIDE);

    rtp_llm::FusedStridedCopyParams params;
    params.add(d_src, d_dst, NROWS, ROW_BYTES, SRC_STRIDE, DST_STRIDE);

    rtp_llm::invokeFusedStridedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto result = deviceToHost(d_dst, NROWS * DST_STRIDE);
    for (size_t r = 0; r < NROWS; ++r)
        for (size_t b = 0; b < ROW_BYTES; ++b)
            ASSERT_EQ(result[r * DST_STRIDE + b], host_src[r * SRC_STRIDE + b]) << "row " << r << " col " << b;

    cudaFree(d_src);
    cudaFree(d_dst);
}

// Compact-to-strided copy: dst has a larger stride than row_bytes.
TEST_F(FusedStridedCopyTest, CompactToStrided) {
    constexpr size_t NROWS      = 4;
    constexpr size_t ROW_BYTES  = 16;
    constexpr size_t SRC_STRIDE = ROW_BYTES;  // compact source
    constexpr size_t DST_STRIDE = 48;         // padded destination

    std::vector<uint8_t> host_src(NROWS * SRC_STRIDE);
    for (size_t i = 0; i < host_src.size(); ++i)
        host_src[i] = static_cast<uint8_t>(i & 0xFF);

    uint8_t* d_src = deviceAlloc(host_src);
    uint8_t* d_dst = deviceAllocZero<uint8_t>(NROWS * DST_STRIDE);

    rtp_llm::FusedStridedCopyParams params;
    params.add(d_src, d_dst, NROWS, ROW_BYTES, SRC_STRIDE, DST_STRIDE);

    rtp_llm::invokeFusedStridedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto result = deviceToHost(d_dst, NROWS * DST_STRIDE);
    for (size_t r = 0; r < NROWS; ++r)
        for (size_t b = 0; b < ROW_BYTES; ++b)
            ASSERT_EQ(result[r * DST_STRIDE + b], host_src[r * SRC_STRIDE + b]) << "row " << r << " col " << b;

    cudaFree(d_src);
    cudaFree(d_dst);
}

// Multiple strided copies in one launch.
TEST_F(FusedStridedCopyTest, MultipleStridedCopies) {
    struct CopySpec {
        size_t nrows, row_bytes, src_stride, dst_stride;
    };
    const std::vector<CopySpec> specs = {
        {4, 16, 32, 16},
        {8, 32, 64, 32},
        {2, 64, 128, 64},
    };

    std::vector<std::vector<uint8_t>> host_srcs(specs.size());
    std::vector<uint8_t*>             d_srcs(specs.size());
    std::vector<uint8_t*>             d_dsts(specs.size());

    for (size_t c = 0; c < specs.size(); ++c) {
        const auto& s = specs[c];
        host_srcs[c].resize(s.nrows * s.src_stride, 0);
        for (size_t r = 0; r < s.nrows; ++r)
            for (size_t b = 0; b < s.row_bytes; ++b)
                host_srcs[c][r * s.src_stride + b] = static_cast<uint8_t>((c * 31 + r * s.row_bytes + b) & 0xFF);

        d_srcs[c] = deviceAlloc(host_srcs[c]);
        d_dsts[c] = deviceAllocZero<uint8_t>(s.nrows * s.dst_stride);
    }

    rtp_llm::FusedStridedCopyParams params;
    for (size_t c = 0; c < specs.size(); ++c) {
        const auto& s = specs[c];
        params.add(d_srcs[c], d_dsts[c], s.nrows, s.row_bytes, s.src_stride, s.dst_stride);
    }

    rtp_llm::invokeFusedStridedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    for (size_t c = 0; c < specs.size(); ++c) {
        const auto& s      = specs[c];
        auto        result = deviceToHost(d_dsts[c], s.nrows * s.dst_stride);
        for (size_t r = 0; r < s.nrows; ++r)
            for (size_t b = 0; b < s.row_bytes; ++b)
                ASSERT_EQ(result[r * s.dst_stride + b], host_srcs[c][r * s.src_stride + b])
                    << "copy " << c << " row " << r << " col " << b;
    }

    for (size_t c = 0; c < specs.size(); ++c) {
        cudaFree(d_srcs[c]);
        cudaFree(d_dsts[c]);
    }
}

// Single-row strided copy (edge case: nrows == 1).
TEST_F(FusedStridedCopyTest, SingleRowCopy) {
    constexpr size_t NROWS      = 1;
    constexpr size_t ROW_BYTES  = 100;
    constexpr size_t SRC_STRIDE = 256;
    constexpr size_t DST_STRIDE = ROW_BYTES;

    std::vector<uint8_t> host_src(NROWS * SRC_STRIDE, 0);
    for (size_t b = 0; b < ROW_BYTES; ++b)
        host_src[b] = static_cast<uint8_t>(b & 0xFF);

    uint8_t* d_src = deviceAlloc(host_src);
    uint8_t* d_dst = deviceAllocZero<uint8_t>(NROWS * DST_STRIDE);

    rtp_llm::FusedStridedCopyParams params;
    params.add(d_src, d_dst, NROWS, ROW_BYTES, SRC_STRIDE, DST_STRIDE);

    rtp_llm::invokeFusedStridedCopy(params, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    auto result = deviceToHost(d_dst, NROWS * DST_STRIDE);
    for (size_t b = 0; b < ROW_BYTES; ++b)
        ASSERT_EQ(result[b], host_src[b]) << "mismatch at byte " << b;

    cudaFree(d_src);
    cudaFree(d_dst);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
