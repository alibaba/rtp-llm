#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"

namespace rtp_llm {

struct RequestInfo {
    std::string frontend_ip;
    std::string dash_ip;
    std::string trace_id;
    std::string request_id;
    std::string source_role;

    bool empty() const {
        return frontend_ip.empty() && dash_ip.empty() && trace_id.empty() && request_id.empty() && source_role.empty();
    }
};

class GenerateInput {
public:
    int inputLength() {
        RTP_LLM_CHECK(input_ids.dim() == 1);
        return input_ids.size(0);
    }

    int promptLength() {
        return inputLength() - prefix_length;
    }

    std::map<int, int> multimodalLengths() const {
        std::map<int, int> lengths;
        if (!multimodal_inputs.has_value() || !multimodal_features.has_value()) {
            return {};
        } else {
            RTP_LLM_CHECK(multimodal_inputs.value().size() == multimodal_features.value().size());
            int mm_num = multimodal_inputs.value().size();
            for (int i = 0; i < mm_num; ++i) {
                lengths[multimodal_inputs.value()[i].mm_type] += multimodal_features.value()[i].size(0);
            }
        }
        return lengths;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateInput {"
                     << "request_id: " << request_id << ", generate_config:" << generate_config->debugString()
                     << ", input_ids: tensor[" << input_ids.numel() << "]"
                     << ", prefix_length:" << prefix_length;
        if (!request_info.empty()) {
            debug_string << ", source_role:" << request_info.source_role << ", frontend_ip:" << request_info.frontend_ip
                         << ", dash_ip:" << request_info.dash_ip << ", trace_id:" << request_info.trace_id
                         << ", source_request_id:" << request_info.request_id;
        }
        debug_string << "}";
        return debug_string.str();
    }

    void updatePrefix(const std::vector<int>& prefix_prompt) {
        // Stage replacements first so malformed backend metadata cannot
        // partially shift a request.
        auto       next_mask        = text_tokens_mask;
        auto       next_locs        = mm_locs;
        auto       next_cache_spans = multimodal_cache_spans;
        const auto added            = static_cast<int64_t>(prefix_prompt.size());
        if (added > 0) {
            const auto token_count = input_ids.numel();
            if (added > std::numeric_limits<int32_t>::max() - token_count) {
                throw std::invalid_argument("system prefix exceeds int32 multimodal coordinates");
            }
            // A P->D ALLOCATE request carries expanded token IDs and the
            // cache-identity sidecar, but intentionally does not carry ViT
            // features, locations or the execution-time text mask.  Decode
            // still has to prepend the same resident system prompt and shift
            // the sidecar.  Raw media without that sidecar is not a cache-only
            // request and must remain fail-closed when its mask is missing.
            const bool has_raw_inputs       = multimodal_inputs && !multimodal_inputs->empty();
            const bool has_cache_spans      = multimodal_cache_spans && !multimodal_cache_spans->empty();
            const bool needs_execution_mask = (multimodal_features && !multimodal_features->empty())
                                              || (mm_locs && mm_locs->numel() != 0)
                                              || (has_raw_inputs && !has_cache_spans);
            if (!text_tokens_mask && needs_execution_mask) {
                throw std::invalid_argument("multimodal system prefix requires a text token mask");
            }
            if (text_tokens_mask) {
                const auto& mask = *text_tokens_mask;
                if (!mask.defined() || !mask.device().is_cpu() || mask.scalar_type() != torch::kInt32 || mask.dim() != 1
                    || mask.numel() != token_count) {
                    throw std::invalid_argument("system prefix text mask does not match the input");
                }
                auto        stable_mask = mask.contiguous();
                const auto* values      = stable_mask.data_ptr<int32_t>();
                for (int64_t pos = 0; pos < token_count; ++pos) {
                    if (values[pos] != 0 && values[pos] != 1) {
                        throw std::invalid_argument("system prefix text mask must be binary");
                    }
                }
                next_mask = torch::cat({torch::ones({added}, mask.options()), mask}, 0);
            }
            if (mm_locs) {
                const auto& locs = *mm_locs;
                if (!locs.defined() || !locs.device().is_cpu() || locs.scalar_type() != torch::kInt32
                    || locs.dim() != 1) {
                    throw std::invalid_argument("system prefix locations must be CPU int32");
                }
                auto        stable_locs = locs.contiguous();
                const auto* values      = stable_locs.data_ptr<int32_t>();
                for (int64_t index = 0; index < locs.numel(); ++index) {
                    if (values[index] < 0 || values[index] >= token_count) {
                        throw std::invalid_argument("system prefix image location is outside the input");
                    }
                }
                next_locs = locs + added;
            }
            if (multimodal_cache_spans) {
                auto    shifted      = std::make_shared<MultimodalCacheSpans>(*multimodal_cache_spans);
                int64_t previous_end = 0;
                for (auto& span : *shifted) {
                    const int64_t end = static_cast<int64_t>(span.offset) + span.row_hashes.size();
                    if (span.offset < previous_end || span.row_hashes.empty() || end > token_count) {
                        throw std::invalid_argument("multimodal cache span is invalid");
                    }
                    span.offset += static_cast<int32_t>(added);
                    previous_end = end;
                }
                next_cache_spans = std::move(shifted);
            }
            // mm_position_ids and mm_extra_input are per-image local coordinates/
            // values, not request token offsets; their contents must not shift.
        }
        auto prefix_tensor     = torch::from_blob(const_cast<int*>(prefix_prompt.data()), {added}, torch::kInt32);
        auto next_ids          = torch::cat({prefix_tensor, input_ids}, 0);
        input_ids              = std::move(next_ids);
        text_tokens_mask       = std::move(next_mask);
        mm_locs                = std::move(next_locs);
        multimodal_cache_spans = std::move(next_cache_spans);
        prefix_length          = prefix_prompt.size();
        if (custom_output_token_position >= 0) {
            custom_output_token_position += prefix_length;
        }
    }

public:
    int64_t                         request_id = 0;
    RequestInfo                     request_info;
    std::shared_ptr<GenerateConfig> generate_config;
    torch::Tensor                   input_ids;
    bool                            need_release_resource = true;
    bool                            fake_query            = false;
    // For multi-modality models
    std::optional<std::vector<MultimodalInput>> multimodal_inputs;
    std::optional<std::vector<torch::Tensor>>   multimodal_features;
    std::optional<torch::Tensor>                text_tokens_mask;  // text part for 1 and multimodal part for 0
    std::optional<torch::Tensor>                mm_locs;           // multimodal input locations
    std::optional<std::vector<torch::Tensor>>   mm_position_ids;
    std::optional<std::vector<torch::Tensor>>   mm_extra_input;
    std::shared_ptr<const MultimodalCacheSpans> multimodal_cache_spans;

    int     prefix_length        = 0;
    int64_t begin_time_us        = 0;
    int64_t global_start_time_us = 0;

    // Auto-TPM QoS priority (task40): 30/40/50/60/70; 0 = not set. TPS metrics
    // tagging only — never used for engine-side scheduling decisions.
    int32_t priority = 0;

    // Selected position in input_ids; -1 means no custom output.
    int custom_output_token_position = -1;

    // Batch grouping params
    int     group_size = 1;
    int64_t group_id   = -1;

    bool isGroup() const {
        return group_id != -1;
    }
};

struct AuxInfo {
    int32_t                      cost_time_us             = 0;
    int32_t                      iter_count               = 0;
    int32_t                      input_len                = 0;
    int32_t                      total_reuse_len          = 0;
    int32_t                      reuse_len                = 0;
    int32_t                      prefix_len               = 0;
    int32_t                      output_len               = 0;
    int32_t                      step_output_len          = 0;
    bool                         pd_sep                   = false;
    int32_t                      first_token_cost_time_us = 0;
    int32_t                      wait_time_us             = 0;
    int32_t                      local_reuse_len          = 0;
    int32_t                      remote_reuse_len         = 0;
    int32_t                      memory_reuse_len         = 0;
    int32_t                      disk_reuse_len           = 0;
    int32_t                      prefill_total_reuse_len  = 0;
    int32_t                      prefill_local_reuse_len  = 0;
    int32_t                      prefill_remote_reuse_len = 0;
    int32_t                      prefill_memory_reuse_len = 0;
    int32_t                      prefill_disk_reuse_len   = 0;
    int32_t                      decode_total_reuse_len   = 0;
    int32_t                      decode_local_reuse_len   = 0;
    int32_t                      decode_remote_reuse_len  = 0;
    int32_t                      decode_memory_reuse_len  = 0;
    int32_t                      decode_disk_reuse_len    = 0;
    int32_t                      speculative_draft_rounds = 0;
    std::vector<int32_t>         speculative_accepted_tokens_per_pos;
    std::string                  generation_prefill_cuda_graph_status = "not_requested";
    std::optional<torch::Tensor> cum_log_probs;
    std::optional<torch::Tensor> all_probs;
    std::optional<torch::Tensor> softmax_probs;
    std::map<int, int>           multimodal_lengths = {};
};

struct PromptLogitsOutput {
    torch::Tensor topk_logprobs;    // [slice_len, K] float32
    torch::Tensor topk_token_ids;   // [slice_len, K] int32
    torch::Tensor target_logprobs;  // [slice_len] float32
    int           start_pos = 0;
    int           end_pos   = 0;
};

class GenerateOutput {
public:
    torch::Tensor output_ids;
    bool          finished;
    AuxInfo       aux_info;
    ErrorInfo     error_info;

    std::optional<torch::Tensor>      hidden_states;
    std::optional<torch::Tensor>      all_hidden_states;
    std::optional<torch::Tensor>      logits;
    std::optional<torch::Tensor>      loss;
    std::optional<PromptLogitsOutput> prompt_logits;
    // Output of the deployment-registered post-layers CustomHandler.
    std::optional<torch::Tensor> custom_output;
};

class GenerateOutputs {
public:
    std::vector<GenerateOutput> generate_outputs;
    int64_t                     request_id;
};

enum class StreamState {
    WAITING,
    LOADING_CACHE,  // GPU blocks allocated, waiting for connector H2D copy
    RUNNING,
    FINISHED
};

inline std::string StreamStateToString(StreamState state) {
    switch (state) {
        case StreamState::WAITING:
            return "WAITING";
        case StreamState::LOADING_CACHE:
            return "LOADING_CACHE";
        case StreamState::RUNNING:
            return "RUNNING";
        case StreamState::FINISHED:
            return "FINISHED";
        default:
            return "Error: Unrecognized Generate State";
    }
}

// 事件集合：外部通过 reportEvent() 投递事件，生命周期方法中统一消费。
// 内部使用 bit flag 组合多个并发事件。
// 生命周期和终止事件一旦设置即保留；CanRun 是一次性调度准入，由调度器在需要重新准入时显式清除。
class StreamEvents {
public:
    enum EventType : uint32_t {
        None               = 0,
        LoadInitiated      = 1 << 0,  // 已尝试加载缓存
        CanRun             = 1 << 1,  // 调度器允许运行
        GenerateDone       = 1 << 2,  // 本地生成完成（RUNNING -> FINISHED）
        Error              = 1 << 3,  // 出错，任何状态 -> FINISHED
        NeedRemoteGenerate = 1 << 4,  // 需要远程生成
        CachePrepared      = 1 << 5,  // KV allocation/cache load completed by the enqueue-time worker
    };

    void append(EventType event) {
        flags_ = static_cast<EventType>(static_cast<uint32_t>(flags_) | static_cast<uint32_t>(event));
    }

    bool has(EventType event) const {
        return (flags_ & event) != 0;
    }

    void clearCanRun() {
        flags_ = static_cast<EventType>(static_cast<uint32_t>(flags_) & ~static_cast<uint32_t>(EventType::CanRun));
    }

    void clearLoadInitiated() {
        flags_ =
            static_cast<EventType>(static_cast<uint32_t>(flags_) & ~static_cast<uint32_t>(EventType::LoadInitiated));
    }

private:
    EventType flags_ = EventType::None;
};

class StreamCacheResource;  // forward declaration

}  // namespace rtp_llm
