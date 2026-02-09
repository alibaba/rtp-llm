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
    int64_t batch_group_id = -1;  // Batch group ID for force batch grouping, -1 means not set
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
    RUNNING,
    PAUSED,
    STOPPED,
    FINISHED,
    REMOTE_RUNNING
};

inline std::string StreamStateToString(StreamState state) {
    switch (state) {
        case StreamState::WAITING:
            return "WAITING";
        case StreamState::RUNNING:
            return "RUNNING";
        case StreamState::PAUSED:
            return "PAUSED";
        case StreamState::STOPPED:
            return "STOPPED";
        case StreamState::FINISHED:
            return "FINISHED";
        case StreamState::REMOTE_RUNNING:
            return "REMOTE_RUNNING";
        default:
            return "Error: Unrecognized Generate State";
    }
}

struct GenerateStatus {
    StreamState status = StreamState::WAITING;
    ErrorInfo   error_info;
};

}  // namespace rtp_llm
