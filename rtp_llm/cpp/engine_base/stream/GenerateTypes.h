#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <vector>
#include <torch/python.h>
#include "rtp_llm/cpp/engine_base/stream/GenerateConfig.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/models/position_ids/PositionIdsGenerator.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"

namespace rtp_llm {

class GenerateInput {
public:
    int inputLength() {
        RTP_LLM_CHECK(input_ids->shape().size() == 1);
        return input_ids->shape()[0];
    }

    int promptLength() {
        return inputLength() - prefix_length;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateInput {"
                     << "request_id: " << request_id << ", generate_config:" << generate_config->debugString()
                     << ", input_ids:" << input_ids->debugStringWithData<int32_t>() << ", lora_id: " << lora_id
                     << ", prefix_length:" << prefix_length << "}";
        return debug_string.str();
    }

    void updatePrefix(const std::vector<int>& prefix_prompt) {
        prefix_length = prefix_prompt.size();
        auto device   = rtp_llm::DeviceFactory::getDefaultDevice();
        input_ids     = device->concat({{rtp_llm::vector2Buffer(prefix_prompt), input_ids}});
    }

public:
    int64_t                         request_id = 0;
    std::shared_ptr<GenerateConfig> generate_config;
    rtp_llm::BufferPtr              input_ids;
    int                             lora_id               = -1;
    bool                            need_release_resource = true;
    bool                            fake_query            = false;
    // For multi-modality models
    std::optional<std::vector<MultimodalInput>> multimodal_inputs;
    std::optional<std::vector<torch::Tensor>>   multimodal_features;
    std::optional<rtp_llm::BufferPtr>           text_tokens_mask;  // text part for 1 and multimodal part for 0
    std::optional<rtp_llm::BufferPtr>           mm_locs;           // multimodal input locations
    std::optional<std::vector<torch::Tensor>>   mm_position_ids;

    int     prefix_length = 0;
    int64_t begin_time_us = 0;
};

struct AuxInfo {
    int32_t                                cost_time_us             = 0;
    int32_t                                iter_count               = 0;
    int32_t                                input_len                = 0;
    int32_t                                total_reuse_len          = 0;
    int32_t                                reuse_len                = 0;
    int32_t                                prefix_len               = 0;
    int32_t                                output_len               = 0;
    int32_t                                fallback_tokens          = 0;
    int32_t                                fallback_times           = 0;
    int32_t                                step_output_len          = 0;
    bool                                   pd_sep                   = false;
    int32_t                                first_token_cost_time_us = 0;
    int32_t                                wait_time_us             = 0;
    int32_t                                local_reuse_len          = 0;
    int32_t                                remote_reuse_len         = 0;
    int32_t                                prefill_total_reuse_len  = 0;
    int32_t                                prefill_local_reuse_len  = 0;
    int32_t                                prefill_remote_reuse_len = 0;
    int32_t                                decode_total_reuse_len   = 0;
    int32_t                                decode_local_reuse_len   = 0;
    int32_t                                decode_remote_reuse_len  = 0;
    std::optional<rtp_llm::ConstBufferPtr> cum_log_probs;
    std::optional<rtp_llm::ConstBufferPtr> all_probs;
    std::optional<rtp_llm::ConstBufferPtr> softmax_probs;
};

class GenerateOutput {
public:
    rtp_llm::ConstBufferPtr output_ids;
    bool                    finished;
    AuxInfo                 aux_info;
    ErrorInfo               error_info;

    std::optional<rtp_llm::ConstBufferPtr> hidden_states;
    std::optional<rtp_llm::ConstBufferPtr> all_hidden_states;
    std::optional<rtp_llm::ConstBufferPtr> logits;
    std::optional<rtp_llm::ConstBufferPtr> loss;
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
    REMOTE_RUNNING,
    LOADING_CACHE
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
