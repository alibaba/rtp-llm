#include "rtp_llm/cpp/core/CopyTypes.h"

namespace rtp_llm {

BatchCopyParams::CopyType BatchCopyParams::get_copy_type(MemoryType dst_type, MemoryType src_type) {
    if (src_type == MEMORY_GPU) {
        return dst_type == MEMORY_GPU ? CopyType::D2D : CopyType::D2H;
    }
    return dst_type == MEMORY_GPU ? CopyType::H2D : CopyType::H2H;
}

BatchCopyParams& BatchCopyParams::reserve(CopyType copy_type, size_t size) {
    RTP_LLM_CHECK_WITH_INFO(copy_type < TYPE_SIZE, "unexpected CopyType %d", copy_type);
    copy_buffers[copy_type].dst_ptr.reserve(size);
    copy_buffers[copy_type].src_ptr.reserve(size);
    copy_buffers[copy_type].sizes.reserve(size);
    return *this;
}

BatchCopyParams& BatchCopyParams::add(void* dst, const void* src, size_t size, CopyType copy_type) {
    RTP_LLM_CHECK_WITH_INFO(copy_type < TYPE_SIZE, "unexpected CopyType %d", copy_type);
    if (size > 0) {
        auto& buffers = copy_buffers[copy_type];
        buffers.dst_ptr.push_back(dst);
        buffers.src_ptr.push_back(src);
        buffers.sizes.push_back(size);
    }
    return *this;
}

}  // namespace rtp_llm
