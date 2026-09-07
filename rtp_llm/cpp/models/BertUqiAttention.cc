#include "rtp_llm/cpp/models/BertUqiAttention.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rtp_llm {

BertUqiBatchMetadata buildBertUqiInputs(const torch::Tensor& tokens,
                                        const torch::Tensor& input_lengths,
                                        int32_t              segment_token_id,
                                        int32_t              separator_token_id) {
    for (const auto& tensor : {tokens, input_lengths}) {
        if (!tensor.defined() || !tensor.device().is_cpu() || tensor.dim() != 1 || tensor.scalar_type() != torch::kInt32
            || !tensor.is_contiguous()) {
            throw std::invalid_argument("BERT UQI tokens and lengths must be contiguous CPU int32 vectors");
        }
    }
    if (segment_token_id < 0 || separator_token_id < 0 || segment_token_id == separator_token_id) {
        throw std::invalid_argument("BERT UQI delimiter IDs must be non-negative and distinct");
    }
    const auto                               batch_size = input_lengths.numel();
    const auto*                              lengths    = input_lengths.data_ptr<int32_t>();
    const auto*                              ids        = tokens.data_ptr<int32_t>();
    const auto                               pinned    = torch::TensorOptions().device(torch::kCPU).pinned_memory(true);
    auto                                     positions = torch::empty({batch_size, 2}, pinned.dtype(torch::kInt64));
    auto*                                    rows      = positions.data_ptr<int64_t>();
    std::vector<std::pair<int32_t, int32_t>> spans;
    spans.reserve(batch_size);
    int64_t offset = 0, mask_size = 0;
    bool    has_profile = false;
    for (int64_t i = 0; i < batch_size; ++i) {
        const int64_t n = lengths[i];
        if (n <= 0 || offset + n > tokens.numel()) {
            throw std::invalid_argument("BERT UQI input length is outside packed tokens");
        }
        mask_size += n * n;
        if (mask_size > std::numeric_limits<int32_t>::max()) {
            throw std::invalid_argument("BERT UQI mask exceeds FlashInfer int32 offset capacity");
        }
        int32_t start = -1, end = -1;
        for (int32_t j = 0; j < n; ++j) {
            if (ids[offset + j] == segment_token_id) {
                if (start >= 0) {
                    throw std::invalid_argument("BERT UQI sequence contains multiple segment markers");
                }
                start = j;
            } else if (start >= 0 && end < 0 && ids[offset + j] == separator_token_id) {
                end = j + 1;
            }
        }
        if (start >= 0 && (end < 0 || end - start == n)) {
            throw std::invalid_argument("BERT UQI profile requires a separator and at least one non-profile token");
        }
        spans.emplace_back(start, end);
        rows[2 * i]     = offset;
        rows[2 * i + 1] = offset + (start < 0 ? 0 : start);
        has_profile |= start >= 0;
        offset += n;
    }
    if (offset != tokens.numel()) {
        throw std::invalid_argument("BERT UQI input lengths do not sum to the packed token count");
    }
    auto mask = torch::empty({has_profile ? mask_size : 0}, pinned.dtype(torch::kBool));
    if (has_profile) {
        auto* data = mask.data_ptr<bool>();
        std::fill_n(data, mask_size, true);
        for (int64_t i = 0; i < batch_size; ++i) {
            const int64_t n         = lengths[i];
            const auto [start, end] = spans[i];
            if (start >= 0) {
                for (int64_t row = 0; row < n; ++row) {
                    if (row < start || row >= end) {
                        std::fill(data + row * n + start, data + row * n + end, false);
                    }
                }
            }
            data += n * n;
        }
    }
    return {std::move(mask), std::move(positions)};
}

}  // namespace rtp_llm
