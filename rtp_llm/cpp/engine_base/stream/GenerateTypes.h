#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"

namespace rtp_llm {

class GenerateInput {
public:
    int inputLength() {
        RTP_LLM_CHECK(input_ids.dim() == 1);
        return input_ids.size(0);
    }

    int promptLength() {
        return inputLength() - prefix_length;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateInput {"
                     << "request_id: " << request_id << ", generate_config:" << generate_config->debugString()
                     << ", input_ids: tensor[" << input_ids.numel() << "]"
                     << ", prefix_length:" << prefix_length << "}";
        return debug_string.str();
    }

    void updatePrefix(const std::vector<int>& prefix_prompt) {
        prefix_length = prefix_prompt.size();
        auto prefix_tensor =
            torch::from_blob(const_cast<int*>(prefix_prompt.data()), {(int64_t)prefix_prompt.size()}, torch::kInt32);
        input_ids = torch::cat({prefix_tensor, input_ids}, 0);
    }

public:
    int64_t                         request_id = 0;
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

    int     prefix_length = 0;
    int64_t begin_time_us = 0;

    // Batch grouping params
    int     batch_group_size = 1;
    int64_t batch_group_id   = -1;  // Batch group ID for force batch grouping, -1 means not set
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
    int32_t                      prefill_total_reuse_len  = 0;
    int32_t                      prefill_local_reuse_len  = 0;
    int32_t                      prefill_remote_reuse_len = 0;
    int32_t                      prefill_memory_reuse_len = 0;
    int32_t                      decode_total_reuse_len   = 0;
    int32_t                      decode_local_reuse_len   = 0;
    int32_t                      decode_remote_reuse_len  = 0;
    int32_t                      decode_memory_reuse_len  = 0;
    std::optional<torch::Tensor> cum_log_probs;
    std::optional<torch::Tensor> all_probs;
    std::optional<torch::Tensor> softmax_probs;
};

class GenerateOutput {
public:
    torch::Tensor output_ids;
    bool          finished;
    AuxInfo       aux_info;
    ErrorInfo     error_info;

    std::optional<torch::Tensor> hidden_states;
    std::optional<torch::Tensor> all_hidden_states;
    std::optional<torch::Tensor> logits;
    std::optional<torch::Tensor> loss;
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

// 事件集合：外部通过 reportEvent() 投递事件，状态机在 moveToNext() 中统一消费。
// 内部使用 bit flag 组合多个并发事件。
// 所有事件均为永久事件：一旦设置即保留，不会被自动清除。
class StreamEvents {
public:
    enum EventType : uint32_t {
        None               = 0,
        LoadInitiated      = 1 << 0,  // 已尝试加载缓存
        CanRun             = 1 << 1,  // 调度器允许运行
        GenerateDone       = 1 << 2,  // 本地生成完成（RUNNING -> FINISHED）
        Error              = 1 << 3,  // 出错，任何状态 -> FINISHED
        NeedRemoteGenerate = 1 << 4,  // 需要远程生成
    };

    void append(EventType event) {
        flags_ = static_cast<EventType>(static_cast<uint32_t>(flags_) | static_cast<uint32_t>(event));
    }

    bool has(EventType event) const {
        return (flags_ & event) != 0;
    }

private:
    EventType flags_ = EventType::None;
};

class StreamCacheResource;  // forward declaration

}  // namespace rtp_llm
