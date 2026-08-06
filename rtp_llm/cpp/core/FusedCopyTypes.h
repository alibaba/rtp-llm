#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace rtp_llm {

// Hard caps keep the parameter structs passed by value to fused-copy kernels bounded.
//
// Sizing rationale:
// - cuda_graph_runner currently needs about 8 contiguous copies plus one strided
//   copy per KV-cache group.
// - PyWrappedModel::forwardMicroBatched is the tightest caller: at most two
//   microbatches each contribute 6 fixed copies plus 4 group copies, for 20 total.
// - A cap of 64 therefore provides roughly 3x headroom and accommodates about
//   30 KV-cache groups. FusedStridedCopyParams remains 3076 bytes at this cap,
//   well below the 32 KiB kernel-parameter limit on supported Volta+ devices.
//
// Before raising either cap, re-check the kernel-parameter limit of the lowest
// supported compute capability and extend the MaxFusedCopies and
// MicroBatchedAccumulationWorstCase tests. These structs must remain POD-like so
// they can be passed to kernels by value. If a caller becomes unbounded, prefer
// chunked launches over dynamic arrays in the kernel signature.
constexpr int MAX_FUSED_D2D_COPIES     = 64;
constexpr int MAX_FUSED_STRIDED_COPIES = 64;

inline void copyParamsAssert(bool value, const std::string& message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}

struct FusedD2DCopyParams {
    const void* src[MAX_FUSED_D2D_COPIES];
    void*       dst[MAX_FUSED_D2D_COPIES];
    size_t      size[MAX_FUSED_D2D_COPIES];
    int         num_copies = 0;

    void add(const void* src_ptr, void* dst_ptr, size_t bytes) {
        copyParamsAssert(num_copies < MAX_FUSED_D2D_COPIES,
                         "FusedD2DCopyParams: num_copies (" + std::to_string(num_copies + 1)
                             + ") exceeds MAX_FUSED_D2D_COPIES (" + std::to_string(MAX_FUSED_D2D_COPIES)
                             + "). Bump the cap in FusedCopyTypes.h after re-checking the capacity.");
        src[num_copies]  = src_ptr;
        dst[num_copies]  = dst_ptr;
        size[num_copies] = bytes;
        ++num_copies;
    }

    void clear() {
        num_copies = 0;
    }
};

struct FusedStridedCopyParams {
    const void* src[MAX_FUSED_STRIDED_COPIES];
    void*       dst[MAX_FUSED_STRIDED_COPIES];
    size_t      num_rows[MAX_FUSED_STRIDED_COPIES];
    size_t      row_bytes[MAX_FUSED_STRIDED_COPIES];
    size_t      src_row_stride[MAX_FUSED_STRIDED_COPIES];
    size_t      dst_row_stride[MAX_FUSED_STRIDED_COPIES];
    int         num_copies = 0;

    void add(const void* src_ptr, void* dst_ptr, size_t rows, size_t row_b, size_t src_stride, size_t dst_stride) {
        copyParamsAssert(num_copies < MAX_FUSED_STRIDED_COPIES,
                         "FusedStridedCopyParams: num_copies (" + std::to_string(num_copies + 1)
                             + ") exceeds MAX_FUSED_STRIDED_COPIES (" + std::to_string(MAX_FUSED_STRIDED_COPIES)
                             + "). Bump the cap in FusedCopyTypes.h after re-checking the capacity.");
        src[num_copies]            = src_ptr;
        dst[num_copies]            = dst_ptr;
        num_rows[num_copies]       = rows;
        row_bytes[num_copies]      = row_b;
        src_row_stride[num_copies] = src_stride;
        dst_row_stride[num_copies] = dst_stride;
        ++num_copies;
    }

    void clear() {
        num_copies = 0;
    }
};

}  // namespace rtp_llm
