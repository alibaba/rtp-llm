#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/TensorDebugUtils.h"
#include "rtp_llm/cpp/utils/StackTrace.h"
#include "rtp_llm/cpp/config/StaticConfig.h"

#include <optional>
#include <functional>
#include <algorithm>
#include <sstream>

namespace rtp_llm {

namespace {

constexpr size_t kMaxTensorLogItems = 64;

std::string tensorDataSuffix(const std::string& tensor_with_data) {
    const auto pos = tensor_with_data.find(", Data(");
    return pos == std::string::npos ? tensor_with_data : tensor_with_data.substr(pos + 2);
}

std::string tensorLogString(const torch::Tensor& tensor, const torch::Tensor& host_snapshot = {}) {
    const torch::Tensor& meta_tensor  = tensor.defined() ? tensor : host_snapshot;
    const torch::Tensor& value_tensor = host_snapshot.defined() ? host_snapshot : tensor;
    if (!meta_tensor.defined()) {
        return tensorDebugString(meta_tensor);
    }
    const auto meta = tensorDebugString(meta_tensor);
    if (!value_tensor.defined() || value_tensor.is_cuda()) {
        return meta;
    }
    std::string data;
    switch (value_tensor.scalar_type()) {
        case torch::kInt32:
            data = tensorDebugStringWithData<int32_t>(value_tensor, kMaxTensorLogItems);
            break;
        case torch::kInt64:
            data = tensorDebugStringWithData<int64_t>(value_tensor, kMaxTensorLogItems);
            break;
        case torch::kBool:
            data = tensorDebugStringWithData<bool>(value_tensor, kMaxTensorLogItems);
            break;
        default:
            return meta;
    }
    return host_snapshot.defined() && tensor.defined() ? meta + ", " + tensorDataSuffix(data) : data;
}

}  // namespace

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
    if (sequence_lengths_plus_1.defined()) {
        debug_string << ", sequence_lengths_plus_1: " << tb(sequence_lengths_plus_1);
    }
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

std::string GptModelInputs::modelInputsLogString() const {
    std::ostringstream os;
    os << "GptModelInputs { ";
    bool first  = true;
    auto append = [&](const char* name, const std::string& value) {
        if (!first) {
            os << ", ";
        }
        first = false;
        os << name << ": " << value;
    };
    auto appendTensor = [&](const char* name, const torch::Tensor& tensor, const torch::Tensor& host_snapshot) {
        append(name, tensorLogString(tensor, host_snapshot));
    };
    auto appendSize = [&](const char* name, size_t value) { append(name, std::to_string(value)); };
    auto appendBool = [&](const char* name, bool value) { append(name, value ? "true" : "false"); };

    append("trace_ids", combineStrings(trace_ids));
    appendTensor("combo_tokens", combo_tokens, combo_tokens_host_for_log);
    appendTensor("input_lengths", input_lengths, input_lengths_host_for_log);
    appendTensor("sequence_lengths", sequence_lengths, sequence_lengths_host_for_log);
    appendTensor("lm_output_indexes", lm_output_indexes, torch::Tensor());
    appendTensor("prefix_lengths", prefix_lengths, prefix_lengths_host_for_log);
    appendTensor("sequence_lengths_plus_1", sequence_lengths_plus_1, torch::Tensor());
    appendTensor("combo_tokens_type_ids", combo_tokens_type_ids, torch::Tensor());
    appendTensor("combo_position_ids", combo_position_ids, torch::Tensor());
    appendTensor("last_hidden_states", last_hidden_states, torch::Tensor());
    appendTensor("attention_mask", attention_mask, torch::Tensor());
    appendTensor("kv_cache_block_id", kv_cache_block_id, torch::Tensor());
    appendTensor("kv_cache_layer_to_group", kv_cache_layer_to_group, torch::Tensor());
    appendTensor("kv_cache_group_types", kv_cache_group_types, torch::Tensor());
    appendTensor("kv_cache_update_mapping", kv_cache_update_mapping, torch::Tensor());
    appendTensor("request_id", request_id, torch::Tensor());
    appendTensor("request_pd_separation", request_pd_separation, torch::Tensor());
    appendSize("kv_block_stride_bytes", kv_block_stride_bytes);
    appendSize("kv_scale_stride_bytes", kv_scale_stride_bytes);
    appendSize("seq_size_per_block", seq_size_per_block);
    appendSize("kernel_seq_size_per_block", kernel_seq_size_per_block);
    appendBool("pd_separation", pd_separation);
    appendBool("decode_entrance", decode_entrance);
    appendBool("use_opaque_kv_cache_store", use_opaque_kv_cache_store);
    appendBool("need_all_logits", need_all_logits);
    appendBool("need_all_hidden_states", need_all_hidden_states);
    appendBool("need_moe_gating", need_moe_gating);
    appendBool("warmup", warmup);
    appendBool("skip_run", skip_run);
    appendBool("is_fake_stream", is_fake_stream);
    appendBool("is_target_verify", is_target_verify);
    os << "}";
    return os.str();
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
