#pragma once
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>
#include <torch/python.h>
#include "absl/status/status.h"
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "maga_transformer/cpp/position_ids_generator/PositionIdsGenerator.h"

namespace ft = fastertransformer;

namespace rtp_llm {

struct MultimodalInput {
// public:
    std::string url;
    int32_t     mm_type = 0;
    MultimodalInput(std::string url, int32_t mm_type = 0): url(url), mm_type(mm_type) {}
};

class MultimodalFeature {
public:
    std::vector<torch::Tensor>   features;
    std::vector<MultimodalInput> inputs;
    ft::BufferPtr                text_tokens_mask; // text part for 1 and multimodal part for 0
    ft::BufferPtr                locs; // multimodal input locations
    ft::BufferPtr                expanded_ids;
    MultimodalFeature() {}
};


class GenerateInput {
public:
    int inputLength() {
        FT_CHECK(input_ids->shape().size() == 1);
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
        auto device     = ft::DeviceFactory::getDefaultDevice();
        input_ids = device->concat({{ft::vector2Buffer(prefix_prompt), input_ids}});
    }

public:
    int64_t                         request_id;
    std::shared_ptr<GenerateConfig> generate_config;
    ft::BufferPtr                   input_ids;

    // For multi-modality models
    // std::optional<MultimodalFeature> multimodal_features;
    std::optional<std::vector<torch::Tensor>>   multimodal_features;
    std::optional<std::vector<MultimodalInput>> multimodal_inputs;
    std::optional<ft::BufferPtr>                text_tokens_mask; // text part for 1 and multimodal part for 0
    std::optional<ft::BufferPtr>                mm_locs; // multimodal input locations
    std::optional<std::vector<ft::BufferPtr>>   mm_position_ids;

    int                             lora_id       = -1;
    int                             prefix_length = 0;
    int64_t                         begin_time_us = 0;

    // config
    bool                            need_release_resource = true;
};

class AuxInfo {
public:
    int                                              cost_time_us   = 0;
    int                                              iter_count     = 0;
    int                                              input_len      = 0;
    int                                              prefix_len     = 0;
    int                                              reuse_len      = 0;
    int                                              output_len     = 0;
    int                                              fallback_tokens = 0;
    int                                              fallback_times  = 0;
    int                                              step_output_len = 0;
    std::optional<ft::ConstBufferPtr>                cum_log_probs;
};

// TODO: add error code.
class ErrorInfo {
public:
    bool        has_error = false;
    std::string error_message;
};

class GenerateOutput {
public:
    ft::ConstBufferPtr              output_ids;
    bool                            finished;
    ErrorInfo                       error_info;
    AuxInfo                         aux_info;

    std::optional<ft::ConstBufferPtr> hidden_states;
    std::optional<ft::ConstBufferPtr> logits;
    std::optional<ft::ConstBufferPtr> loss;
};

class GenerateOutputs {
public:
    std::vector<GenerateOutput> generate_outputs;
    int64_t                     request_id;
};

enum class GenerateState {
    WAITING,
    RUNNING,
    PAUSED,
    STOPPED,
    FINISHED,
};

struct GenerateStatus {
    GenerateState    status = GenerateState::WAITING;
    absl::StatusCode error_code;
    std::string      error_info;
};

void registerMultimodalInput(const py::module& m);

}  // namespace rtp_llm
