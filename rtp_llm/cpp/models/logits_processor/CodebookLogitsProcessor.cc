#include "rtp_llm/cpp/models/logits_processor/CodebookLogitsProcessor.h"

#include <algorithm>
#include <limits>
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

CodebookLogitsProcessor::CodebookLogitsProcessor(const std::vector<std::vector<int64_t>>& groups,
                                                 size_t                                   vocab_size,
                                                 size_t                                   batch_size):
    CodebookLogitsProcessor(createMasks(groups, vocab_size), batch_size) {}

CodebookLogitsProcessor::CodebookLogitsProcessor(torch::Tensor masks, size_t batch_size):
    vocab_size_(0), batch_size_(batch_size), masks_(std::move(masks)) {
    RTP_LLM_CHECK_WITH_INFO(masks_.defined() && masks_.dim() == 2 && masks_.scalar_type() == torch::kBool
                                && masks_.is_contiguous() && masks_.size(0) > 0 && masks_.size(1) > 0,
                            "codebook masks must be a nonempty contiguous bool matrix");
    RTP_LLM_CHECK_WITH_INFO(batch_size_ > 0, "codebook batch size must not be zero");
    vocab_size_ = masks_.size(1);
}

std::optional<std::string> CodebookLogitsProcessor::validateGroups(const std::vector<std::vector<int64_t>>& groups,
                                                                   const std::vector<int64_t>&              ids,
                                                                   int64_t eos_token_id) {
    if (groups.empty()) {
        return std::nullopt;
    }
    if (ids.empty()) {
        return "codebook groups require output vocabulary pruning";
    }
    if (groups.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return "too many codebook levels";
    }
    for (const auto& group : groups) {
        if (group.empty() || !std::is_sorted(group.begin(), group.end())
            || std::adjacent_find(group.begin(), group.end()) != group.end()) {
            return "codebook groups must be nonempty, sorted and unique";
        }
        for (auto compact_id : group) {
            if (compact_id < 0 || static_cast<size_t>(compact_id) >= ids.size()) {
                return "codebook token is outside the compact output vocabulary";
            }
            if (ids[compact_id] == eos_token_id) {
                return "codebook groups must not contain EOS";
            }
        }
    }
    return std::nullopt;
}

torch::Tensor CodebookLogitsProcessor::createMasks(const std::vector<std::vector<int64_t>>& groups, size_t vocab_size) {
    RTP_LLM_CHECK_WITH_INFO(!groups.empty(), "codebook token groups must not be empty");
    RTP_LLM_CHECK_WITH_INFO(vocab_size > 0, "codebook vocabulary must not be empty");

    auto masks    = torch::ones({static_cast<int64_t>(groups.size()), static_cast<int64_t>(vocab_size)},
                             torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    auto accessor = masks.accessor<bool, 2>();
    for (size_t level = 0; level < groups.size(); ++level) {
        RTP_LLM_CHECK_WITH_INFO(!groups[level].empty(), "codebook token group must not be empty");
        for (const int64_t token_id : groups[level]) {
            RTP_LLM_CHECK_WITH_INFO(token_id >= 0 && static_cast<size_t>(token_id) < vocab_size,
                                    "codebook token id is outside compact vocabulary");
            accessor[level][token_id] = false;
        }
    }
    return masks;
}

torch::Tensor CodebookLogitsProcessor::maskFor(const torch::Device& device) {
    // Engine-created processors already reference the shared device tensor.
    // Standalone callers can supply host masks and upload on first use.
    if (masks_.device() != device) {
        masks_ = masks_.to(device);
    }
    return masks_;
}

std::optional<ErrorInfo>
CodebookLogitsProcessor::process(const SamplerInputs& inputs, size_t start_idx, size_t finish_idx) {
    if (inputs.logits.dim() != 2 || static_cast<size_t>(inputs.logits.size(1)) != vocab_size_) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "codebook logits shape does not match vocabulary");
    }
    if (start_idx > finish_idx || finish_idx > static_cast<size_t>(inputs.logits.size(0))
        || finish_idx - start_idx != batch_size_) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "codebook logits interval does not match batch size");
    }
    if (committed_steps_ >= static_cast<size_t>(masks_.size(0))) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "codebook logits processor has no remaining level");
    }

    auto logits = inputs.logits.narrow(0, start_idx, batch_size_);
    auto mask   = maskFor(inputs.logits.device()).select(0, committed_steps_).unsqueeze(0);
    if (!inputs.finished_mask.defined()) {
        logits.masked_fill_(mask, -std::numeric_limits<float>::infinity());
        return std::nullopt;
    }
    // Leave completed rows untouched so MultiSeq's EOS-only distribution survives.
    // Early stopping can leave fewer legal candidates than the requested beam width.
    // Preserve existing top-k behavior: extra beams may select masked positions
    // (e.g. [10, 10] after stop token 10). No per-step candidate check or logging.
    if (!inputs.finished_mask.device().is_cpu() || inputs.finished_mask.scalar_type() != torch::kBool
        || inputs.finished_mask.dim() != 1 || !inputs.finished_mask.is_contiguous()
        || static_cast<size_t>(inputs.finished_mask.numel()) < finish_idx) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "codebook finished mask must be a host bool vector");
    }
    const auto* finished = inputs.finished_mask.data_ptr<bool>() + start_idx;
    size_t      row      = 0;
    while (row < batch_size_) {
        if (finished[row]) {
            ++row;
            continue;
        }
        const auto first = row;
        while (row < batch_size_ && !finished[row]) {
            ++row;
        }
        logits.narrow(0, first, row - first).masked_fill_(mask, -std::numeric_limits<float>::infinity());
    }
    return std::nullopt;
}

void CodebookLogitsProcessor::updateMultiSeqStatus(const std::vector<int>& src_batch_indices) {
    RTP_LLM_CHECK_WITH_INFO(!src_batch_indices.empty(), "codebook beam mapping must not be empty");
    for (const int src_batch_idx : src_batch_indices) {
        RTP_LLM_CHECK_WITH_INFO(src_batch_idx >= 0 && static_cast<size_t>(src_batch_idx) < batch_size_,
                                "codebook beam mapping contains an invalid parent index");
    }
    batch_size_ = src_batch_indices.size();
}

std::optional<ErrorInfo> CodebookLogitsProcessor::updateStatus(const torch::Tensor& new_tokens,
                                                               int32_t              num_new_tokens) {
    if (new_tokens.dim() != 2) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "codebook committed tokens must be a two-dimensional tensor");
    }
    if (num_new_tokens < 0) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "codebook committed token count must not be negative");
    }
    const auto next_steps = committed_steps_ + static_cast<size_t>(num_new_tokens);
    if (next_steps > static_cast<size_t>(masks_.size(0))) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "codebook committed token count exceeds available levels");
    }
    committed_steps_ = next_steps;
    return std::nullopt;
}

std::optional<int64_t> CodebookLogitsProcessor::committedOutputLen() const {
    return static_cast<int64_t>(committed_steps_);
}

bool CodebookLogitsProcessor::isStateful() const {
    return true;
}

bool CodebookLogitsProcessor::supportsNormalAsyncDeviceState() const {
    return false;
}

}  // namespace rtp_llm
