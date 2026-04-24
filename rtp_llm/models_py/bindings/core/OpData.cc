#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

#include <optional>
#include <functional>
#include <algorithm>
#include <sstream>

namespace rtp_llm {

std::string combineStrings(const std::vector<std::string>& vec) {
    std::string result = "\" ";
    for (const auto& s : vec) {
        result += s + ", ";
    }
    result += "\"";
    return result;
}

OpException::OpException(const OpStatus& status): status_(status) {
    std::stringstream ss;
    ss << "OpException[" << (int32_t)status_.error_type << "]: " << status_.error_message << std::endl;
    RTP_LLM_LOG_INFO("%s", ss.str().c_str());
    const auto stack = rtp_llm::getStackTrace();
    RTP_LLM_STACKTRACE_LOG_INFO("%s", stack.c_str());
    ss << stack;
    detail_str_ = ss.str();
    if (StaticConfig::user_ft_core_dump_on_exception) {
        fflush(stdout);
        fflush(stderr);
        abort();
    }
}

std::string GptModelInputs::debugString(bool force) const {
    if (!Logger::getEngineLogger().isDebugMode() && !force) {
        return "";
    }
    auto              tb = [](const torch::Tensor& t) -> std::string { return tensorDebugString(t); };
    std::stringstream debug_string;
    debug_string << "GptModelInputs { "
                 << "trace_ids: " << combineStrings(trace_ids) << ", combo_tokens: " << tb(combo_tokens)
                 << ", input_lengths: " << tb(input_lengths) << ", sequence_lengths: " << tb(sequence_lengths)
                 << ", prefix_lengths: " << tb(prefix_lengths);
    if (combo_position_ids.defined()) {
        debug_string << ", combo_position_ids: " << tb(combo_position_ids);
    }
    if (kv_cache_kernel_block_id.defined()) {
        debug_string << ", kv_cache_kernel_block_id: " << tb(kv_cache_kernel_block_id);
    }
    if (kv_cache_block_id.defined()) {
        debug_string << ", kv_cache_block_id: " << tb(kv_cache_block_id);
    }
    if (attention_mask.defined()) {
        debug_string << ", attention_mask: " << tb(attention_mask);
    }
    if (request_id.defined()) {
        debug_string << ", request_id: " << tb(request_id);
    }
    if (request_pd_separation.defined()) {
        debug_string << ", request_pd_separation: " << tb(request_pd_separation);
    }
    if (cache_keys.defined()) {
        debug_string << ", cache_keys: " << tb(cache_keys);
    }
    debug_string << ", kv_block_stride_bytes: " << kv_block_stride_bytes;
    debug_string << ", pd_separation: " << pd_separation;
    debug_string << "}";
    return debug_string.str();
}

BatchCopyParams::CopyType BatchCopyParams::get_copy_type(MemoryType dst_type, MemoryType src_type) {
    if (src_type == MEMORY_GPU) {
        if (dst_type == MEMORY_GPU) {
            return CopyType::D2D;
        } else {
            return CopyType::D2H;
        }
    } else {
        if (dst_type == MEMORY_GPU) {
            return CopyType::H2D;
        } else {
            return CopyType::H2H;
        }
    }
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
