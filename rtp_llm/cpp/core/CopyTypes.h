#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <torch/torch.h>

#include "rtp_llm/cpp/core/FusedCopyTypes.h"
#include "rtp_llm/cpp/core/MemoryTypes.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct CopyParams {
    const torch::Tensor& dst;
    const torch::Tensor& src;
    bool                 overlapped = false;
    bool                 async      = true;

    void check() const {
        RTP_LLM_CHECK_WITH_INFO(src.scalar_type() == dst.scalar_type(), "copy dst and src need has same type.");
        RTP_LLM_CHECK_WITH_INFO(
            src.nbytes() == dst.nbytes(), "src and dst copy size mismatch: %zu vs %zu", src.nbytes(), dst.nbytes());
    }
};

struct MultiCopyParams {
    std::vector<torch::Tensor> multi_dst;
    std::vector<torch::Tensor> multi_src;

    // CUDA split-KV scatter/gather path. When enabled, each block is staged
    // as interleaved KV/scale regions before the device kernel fans it out.
    int    split_kv_layer_num          = 0;
    size_t split_kv_cache_stride_bytes = 0;
    size_t split_kv_scale_stride_bytes = 0;
};

struct BatchedMemoryCopyTile {
    void*       dst   = nullptr;
    const void* src   = nullptr;
    size_t      bytes = 0;
};

struct BatchedMemoryCopyParams {
    std::vector<BatchedMemoryCopyTile> tiles;
    int                                device_index = -1;
};

enum class StagedMemoryCopyDirection {
    H2D = 0,
    D2H = 1,
};

struct StagedMemoryCopyTile {
    void*  gpu         = nullptr;
    size_t host_offset = 0;
    size_t bytes       = 0;
};

struct StagedMemoryCopyHostSegment {
    void*  host        = nullptr;
    size_t host_offset = 0;
    size_t bytes       = 0;
};

struct StagedMemoryCopyParams {
    void*                                    host_base  = nullptr;
    size_t                                   host_bytes = 0;
    std::vector<StagedMemoryCopyHostSegment> host_segments;
    std::vector<StagedMemoryCopyTile>        tiles;
    int                                      device_index = -1;
    StagedMemoryCopyDirection                direction    = StagedMemoryCopyDirection::H2D;
};

struct StagedMemoryCopyScratch {
    void*  host_staging       = nullptr;
    size_t host_capacity      = 0;
    void*  device_staging     = nullptr;
    size_t device_capacity    = 0;
    void*  device_ptrs        = nullptr;
    void*  device_offsets     = nullptr;
    void*  device_sizes       = nullptr;
    size_t meta_capacity      = 0;
    int    device_index       = -1;
};

struct MultiMergeCopyParams {
    void*               dst_ptr;
    std::vector<void*>  src_ptrs;
    std::vector<size_t> copy_size;
    std::vector<size_t> dst_offsets;
};

struct BatchCopyParams {
    enum CopyType : uint32_t {
        D2H = 0,
        H2D = 1,
        D2D = 2,
        H2H = 3,
        TYPE_SIZE
    };

    struct BatchCopyBuffers {
        std::vector<void*>       dst_ptr;
        std::vector<const void*> src_ptr;
        std::vector<uint64_t>    sizes;
    };

    BatchCopyBuffers copy_buffers[TYPE_SIZE];
    bool             overlapped = false;

    BatchCopyParams& set_overlapped(bool value) {
        overlapped = value;
        return *this;
    }

    static CopyType  get_copy_type(MemoryType dst_type, MemoryType src_type);
    BatchCopyParams& reserve(CopyType copy_type, size_t size);
    BatchCopyParams& add(void* dst, const void* src, size_t size, CopyType copy_type);
};

}  // namespace rtp_llm
