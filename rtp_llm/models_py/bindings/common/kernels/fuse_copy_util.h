#pragma once
#include <cstddef>
#include <stdexcept>

namespace rtp_llm {

// NOTE: Hardcoded limits for fused copies. It is enough for most cases. If you need more, please increase the limits.
static constexpr int MAX_FUSED_D2D_COPIES     = 16;
static constexpr int MAX_FUSED_STRIDED_COPIES = 16;

inline void copyParamsAssert(bool value, const std::string& msg) {
    if (!value) {
        throw std::runtime_error(msg);
    }
}

struct FusedD2DCopyParams {
    const void* src[MAX_FUSED_D2D_COPIES];
    void*       dst[MAX_FUSED_D2D_COPIES];
    size_t      size[MAX_FUSED_D2D_COPIES];
    int         num_copies = 0;

    void add(const void* src_ptr, void* dst_ptr, size_t bytes) {
        copyParamsAssert(num_copies < MAX_FUSED_D2D_COPIES,
                         "FusedD2DCopyParams: num_copies exceeds MAX_FUSED_D2D_COPIES");
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
                         "FusedStridedCopyParams: num_copies exceeds MAX_FUSED_STRIDED_COPIES");
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
