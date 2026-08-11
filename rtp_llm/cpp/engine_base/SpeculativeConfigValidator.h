#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "rtp_llm/cpp/config/ModelConfig.h"

namespace rtp_llm {

inline size_t effectiveInputVocabSize(const ModelConfig& model_config) {
    if (model_config.input_vocab_size < 0) {
        throw std::invalid_argument("model input_vocab_size must not be negative");
    }
    if (model_config.embedding_size < 0) {
        throw std::invalid_argument("model embedding_size must not be negative");
    }
    int64_t input_vocab_size =
        model_config.input_vocab_size == 0 ? model_config.vocab_size : model_config.input_vocab_size;
    if (model_config.embedding_size > 0) {
        input_vocab_size = std::min(input_vocab_size, model_config.embedding_size);
    }
    if (input_vocab_size <= 0) {
        throw std::invalid_argument("model input vocabulary must be positive");
    }
    return static_cast<size_t>(input_vocab_size);
}

}  // namespace rtp_llm
