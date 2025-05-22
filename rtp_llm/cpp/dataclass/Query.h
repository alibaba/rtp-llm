#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <torch/python.h>
#include "rtp_llm/cpp/dataclass/GenerateConfig.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/position_ids_generator/PositionIdsGenerator.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"



namespace rtp_llm {

struct MMPreprocessConfig {
    int32_t width     = -1;
    int32_t height    = -1;
    int32_t min_pixels = -1;
    int32_t max_pixels = -1;
    int32_t fps       = -1;
    int32_t min_frames = -1;
    int32_t max_frames = -1;
    MMPreprocessConfig(int32_t width = -1, int32_t height = -1, int32_t min_pixels = -1, int32_t max_pixels = -1, int32_t fps = -1, int32_t min_frames = -1, int32_t max_frames = -1):
        width(width), height(height), min_pixels(min_pixels), max_pixels(max_pixels), fps(fps), min_frames(min_frames), max_frames(max_frames) {}
};

struct MultimodalInput {
// public:
    std::string url;
    torch::Tensor      tensor               = torch::empty({0});
    int32_t            mm_type              = 0;
    MMPreprocessConfig mm_preprocess_config = MMPreprocessConfig();
    MultimodalInput(std::string url, torch::Tensor t, int32_t mm_type = 0, int32_t width = -1, int32_t height = -1, int32_t min_pixels = -1, int32_t max_pixels = -1, int32_t fps = -1, int32_t min_frames = -1, int32_t max_frames = -1):
        url(url), tensor(t), mm_type(mm_type), mm_preprocess_config(MMPreprocessConfig(width, height, min_pixels, max_pixels, fps, min_frames, max_frames)) {}
    MultimodalInput(std::string url):
        url(url), tensor(torch::empty(0)) {}
};

struct MultimodalOutput {
    std::vector<torch::Tensor> mm_features = {};
    std::optional<std::vector<torch::Tensor>> mm_position_ids = std::nullopt;
};

class MultimodalFeature {
public:
    std::vector<torch::Tensor>   features;
    std::vector<MultimodalInput> inputs;
    rtp_llm::BufferPtr                text_tokens_mask; // text part for 1 and multimodal part for 0
    rtp_llm::BufferPtr                locs; // multimodal input locations
    rtp_llm::BufferPtr                expanded_ids;
    MultimodalFeature() {}
};


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
        prefix_length   = prefix_prompt.size();
        auto device     = rtp_llm::DeviceFactory::getDefaultDevice();
        input_ids = device->concat({{rtp_llm::vector2Buffer(prefix_prompt), input_ids}});
    }

public:
    int64_t                         request_id              = 0;
    std::shared_ptr<GenerateConfig> generate_config;
    rtp_llm::BufferPtr                   input_ids;
    int                             lora_id                 = -1;
    bool                            need_release_resource   = true;
    bool                            fake_query              = false;
    // For multi-modality models
    std::optional<std::vector<MultimodalInput>> multimodal_inputs;
    std::optional<std::vector<torch::Tensor>>   multimodal_features;
    std::optional<rtp_llm::BufferPtr>                text_tokens_mask;   // text part for 1 and multimodal part for 0
    std::optional<rtp_llm::BufferPtr>                mm_locs;            // multimodal input locations
    std::optional<std::vector<torch::Tensor>>   mm_position_ids;

    int                             prefix_length = 0;
    int64_t                         begin_time_us = 0;
};

class AuxInfo {
public:
    int                                              cost_time_us   = 0;
    int                                              first_token_cost_time_us = 0;
    int                                              wait_time_us = 0;
    int                                              iter_count     = 0;
    int                                              input_len      = 0;
    int                                              prefix_len     = 0;
    int                                              reuse_len      = 0;
    int                                              output_len     = 0;
    int                                              fallback_tokens = 0;
    int                                              fallback_times  = 0;
    int                                              step_output_len = 0;
    bool                                             pd_sep          = false;
    std::optional<rtp_llm::ConstBufferPtr>                cum_log_probs;
    std::optional<rtp_llm::ConstBufferPtr>                all_probs;
    std::optional<rtp_llm::ConstBufferPtr>                softmax_probs;
};


class GenerateOutput {
public:
    rtp_llm::ConstBufferPtr              output_ids;
    bool                            finished;
    AuxInfo                         aux_info;
    ErrorInfo                       error_info;

    std::optional<rtp_llm::ConstBufferPtr> hidden_states;
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

void registerMultimodalInput(const py::module& m);

}  // namespace rtp_llm
