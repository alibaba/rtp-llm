#pragma once
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"

#include <assert.h>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace ft = fastertransformer;

namespace rtp_llm {

class GenerateInput {
public:
    int inputLength() {
        assert(input_ids->shape().size() == 1);
        return input_ids->shape()[0];
    }

    int promptLength() {
        return inputLength() - prefix_length;
    }

    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "GenerateInput {"
                     << "request_id: " << request_id << ", generate_config:" << generate_config->debugString()
                     << ", input_ids:" << input_ids->debugString() << ", prefix_length:" << prefix_length << "}";
        return debug_string.str();
    }

    void updatePrefix(const std::vector<int>& prefix_prompt) {
        prefix_length   = prefix_prompt.size();
        auto device     = ft::DeviceFactory::getDevice(ft::DeviceType::Cuda);
        auto new_input  = device->allocateBuffer({ft::DataType::TYPE_INT32, {(size_t)prefix_length + (size_t)inputLength()}, ft::AllocationType::HOST}, {});
        // TODO(xinfei.sxf) fix this
        auto buffer = ft::vector2Buffer(prefix_prompt);
        auto bufferPtr = convertBuffer2Ptr(buffer);
        ft::bufferConcat(bufferPtr, input_ids, new_input);
        input_ids = std::move(new_input);
    }

public:
    int64_t                         request_id;
    std::shared_ptr<GenerateConfig> generate_config;
    ft::BufferPtr                   input_ids;
    std::optional<ft::BufferPtr>    input_embeddings;  // For multi-modality models
    std::optional<int>              lora_id       = -1;
    int                             prefix_length = 0;
};

class AuxInfo {
public:
    int                                              cost_time_ms   = 0;
    int                                              iter_count     = 0;
    int                                              input_len      = 0;
    int                                              prefix_len     = 0;
    int                                              reuse_len      = 0;
    int                                              output_len     = 0;
    std::optional<std::shared_ptr<const ft::Buffer>> cum_log_probs;
};

// TODO: add error code.
class ErrorInfo {
public:
    bool        has_error = false;
    std::string error_message;
};

class GenerateOutput {
public:
    std::shared_ptr<const ft::Buffer> output_ids;
    bool                              finished;
    ErrorInfo                         error_info;
    AuxInfo                           aux_info;

    std::optional<std::shared_ptr<const ft::Buffer>> hidden_states;
    std::optional<std::shared_ptr<const ft::Buffer>> logits;
    std::optional<std::shared_ptr<const ft::Buffer>> loss;
};

enum class GenerateState {
    WAITING,
    RUNNING,
    STOPPED,
    FINISHED,
};

struct GenerateStatus {
    GenerateState status = GenerateState::WAITING;
    std::string   error_info;
};

}  // namespace rtp_llm
