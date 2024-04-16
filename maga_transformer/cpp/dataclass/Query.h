#pragma once
#include "maga_transformer/cpp/dataclass/GenerateConfig.h"
#include "src/fastertransformer/core/Buffer.h"
#include <assert.h>
#include <cstdint>
#include <optional>
#include <sstream>
#include <string>

namespace ft = fastertransformer;

namespace rtp_llm {

struct MedusaState {};

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
    int                                              cost_time_ms;
    int                                              iter_count;
    int                                              input_len;
    int                                              prefix_len;
    int                                              reuse_len;
    int                                              output_len;
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
