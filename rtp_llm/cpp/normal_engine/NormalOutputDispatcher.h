#pragma once

#include <functional>
#include <optional>

#include <torch/all.h>
#include <utility>
#include <vector>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

std::optional<ErrorInfo> collectStreamSamplerError(const SamplerOutput& sampler_output,
                                                   const torch::Tensor& success_cpu,
                                                   int                  batch_idx_in,
                                                   int                  cur_batch_size);

class NormalOutputDispatcher {
public:
    explicit NormalOutputDispatcher(std::vector<int64_t> output_vocab_ids = {}, bool warm_up = false):
        output_vocab_ids_(std::move(output_vocab_ids)), warm_up_(warm_up) {}

    absl::Status dispatch(const StreamGroups& stream_groups, const MergedOutput& merge_outputs) const;

    // Test seam (nullptr in production): invoked inside dispatchSingleStream's guarded
    // prompt-scoring/loss block so tests can inject e.g. a c10::OutOfMemoryError for a
    // specific stream without exhausting real memory. Modeled on
    // NormalExecutor::test_model_factory.
    static std::function<void(const GenerateStreamPtr&)> test_scoring_failure_hook;

private:
    bool restoreCurrentTokenIds(const GenerateStreamPtr& stream,
                                torch::Tensor&           batch_token_ids,
                                torch::Tensor&           current_token_ids,
                                size_t                   token_position) const;

    // Marks the stream failed for a scoring-stage out-of-memory. Returns false when the
    // OOM must propagate instead (warm-up), so the caller rethrows.
    bool failStreamOnScoringOom(const GenerateStreamPtr& stream, const char* what) const;

    void dispatchSingleStream(GenerateStreamPtr    stream,
                              const MergedOutput&  merge_outputs,
                              int                  batch_idx_in,
                              int                  batch_idx_out,
                              int                  token_offset,
                              int&                 all_logits_row_offset,
                              bool                 return_all_probs,
                              const torch::Tensor& new_tokens_all,
                              const torch::Tensor& token_ids_cpu,
                              const torch::Tensor& success_cpu) const;

private:
    std::vector<int64_t> output_vocab_ids_;
    bool                 warm_up_ = false;
};

}  // namespace rtp_llm
