#pragma once

#include <cstdint>
#include <sstream>
#include <string>

#include "absl/status/status.h"

namespace rtp_llm {

inline absl::Status
validateTextTokensMaskLength(int64_t stream_id, const int* text_tokens_mask, int text_tokens_mask_length, int length) {
    if (text_tokens_mask != nullptr && text_tokens_mask_length != length) {
        std::ostringstream error_msg;
        error_msg << "stream [" << stream_id << "] text_tokens_mask length " << text_tokens_mask_length
                  << " does not match input length " << length;
        return absl::InvalidArgumentError(error_msg.str());
    }
    return absl::OkStatus();
}

// Word-id validation for the normal engine. A zero mask marks a feature row
// whose word id is not used. An absent mask means all text. A vocabulary bound
// of 0 means "not configured" (ModelConfig default) and disables only the
// upper-bound check. Negative text IDs are invalid even without that bound.
inline absl::Status validateEmbeddingIdRanges(int64_t    stream_id,
                                              const int* token_ids,
                                              const int* text_tokens_mask,
                                              int        text_tokens_mask_length,
                                              int        length,
                                              int        input_vocab_size) {
    auto mask_status = validateTextTokensMaskLength(stream_id, text_tokens_mask, text_tokens_mask_length, length);
    if (!mask_status.ok()) {
        return mask_status;
    }
    for (int index = 0; index < length; ++index) {
        const bool is_text_token = text_tokens_mask == nullptr || text_tokens_mask[index] != 0;
        if (is_text_token) {
            const int token_id = token_ids[index];
            if (token_id < 0 || (input_vocab_size > 0 && token_id >= input_vocab_size)) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream_id << "] token_id " << token_id << " exceed vocab_size "
                          << input_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
        }
    }
    return absl::OkStatus();
}

}  // namespace rtp_llm
