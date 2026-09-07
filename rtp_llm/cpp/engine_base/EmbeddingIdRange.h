#pragma once

#include <algorithm>
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

inline void
normalizeMaskedTokenTypeIds(int* token_type_ids, const int* text_tokens_mask, int text_tokens_mask_length, int length) {
    if (token_type_ids == nullptr || text_tokens_mask == nullptr) {
        return;
    }
    const int checked_length = std::min(text_tokens_mask_length, length);
    for (int index = 0; index < checked_length; ++index) {
        if (text_tokens_mask[index] == 0) {
            token_type_ids[index] = 0;
        }
    }
}

// Shared id-range contract for the two engines that feed the embedding kernels.
// The kernels index the word and token-type tables without device-side guards,
// so those ids must be validated on the host before the forward. Position ids
// are generated and bounded by each caller's position-id path.
//
// A zero mask exempts only the word id: the kernel still reads position and
// token-type tables before applying the mask. Callers must normalize any image-
// row token-type sentinel before validation. An absent mask means all text. A
// bound of 0 means "not configured" (ModelConfig default) and disables that
// check.
inline absl::Status validateEmbeddingIdRanges(int64_t    stream_id,
                                              const int* token_ids,
                                              const int* token_type_ids,
                                              const int* text_tokens_mask,
                                              int        text_tokens_mask_length,
                                              int        length,
                                              int        input_vocab_size,
                                              int        type_vocab_size) {
    auto mask_status = validateTextTokensMaskLength(stream_id, text_tokens_mask, text_tokens_mask_length, length);
    if (!mask_status.ok()) {
        return mask_status;
    }
    for (int index = 0; index < length; ++index) {
        const bool is_text_token = text_tokens_mask == nullptr || text_tokens_mask[index] != 0;
        if (is_text_token && input_vocab_size > 0) {
            const int token_id = token_ids[index];
            if (token_id < 0 || token_id >= input_vocab_size) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream_id << "] token_id " << token_id << " exceed vocab_size "
                          << input_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
        }
        if (token_type_ids != nullptr && type_vocab_size > 0) {
            const int token_type_id = token_type_ids[index];
            if (token_type_id < 0 || token_type_id >= type_vocab_size) {
                std::ostringstream error_msg;
                error_msg << "stream [" << stream_id << "] token_type_id " << token_type_id
                          << " exceed type_vocab_size " << type_vocab_size;
                return absl::InvalidArgumentError(error_msg.str());
            }
        }
    }
    return absl::OkStatus();
}

inline absl::Status
validatePositionIdRange(int64_t stream_id, int sequence_length, int position_bias, int position_embedding_count) {
    if (sequence_length < 0 || position_bias < 0 || position_embedding_count <= 0
        || position_bias > position_embedding_count || sequence_length > position_embedding_count - position_bias) {
        std::ostringstream error_msg;
        error_msg << "stream [" << stream_id << "] sequence_length " << sequence_length << " + position_bias "
                  << position_bias << " exceed position_embedding_count " << position_embedding_count;
        return absl::InvalidArgumentError(error_msg.str());
    }
    return absl::OkStatus();
}

inline int positionEmbeddingRowLimit(int max_seq_len, int64_t position_embedding_rows) {
    return position_embedding_rows > 0 && position_embedding_rows < max_seq_len ?
               static_cast<int>(position_embedding_rows) :
               max_seq_len;
}

inline int positionIdBias(int64_t position_ids_style, int pad_token_id) {
    return position_ids_style == 1 ? pad_token_id + 1 : 0;
}

// Keep sentinel normalization and range validation in one ordered operation.
// The length check must run first so malformed masks cannot partially mutate
// the token-type buffer before the request is rejected.
inline absl::Status normalizeAndValidateEmbeddingIds(int64_t    stream_id,
                                                     const int* token_ids,
                                                     int*       token_type_ids,
                                                     const int* text_tokens_mask,
                                                     int        text_tokens_mask_length,
                                                     int        length,
                                                     int        input_vocab_size,
                                                     int        type_vocab_size) {
    auto mask_status = validateTextTokensMaskLength(stream_id, text_tokens_mask, text_tokens_mask_length, length);
    if (!mask_status.ok()) {
        return mask_status;
    }
    normalizeMaskedTokenTypeIds(token_type_ids, text_tokens_mask, text_tokens_mask_length, length);
    return validateEmbeddingIdRanges(stream_id,
                                     token_ids,
                                     token_type_ids,
                                     text_tokens_mask,
                                     text_tokens_mask_length,
                                     length,
                                     input_vocab_size,
                                     type_vocab_size);
}

}  // namespace rtp_llm
