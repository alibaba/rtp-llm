#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

// Glue between MtpExecutor and MTP-aware per-stream logits processors; emits compact
// packed allow-masks plus the target logits rows they constrain.
//
// Single-flight and non-reentrant: LaunchResult tensors are views into reusable
// internal buffers. The caller must finish consuming one result, including GPU work
// that reads its views, before calling run() again on the same runner.
class SpecLogitsVerifyRunner {
public:
    struct ActiveProcessor {
        BaseLogitsProcessorPtr processor;
        size_t                 stream_idx = 0;
    };

    struct LaunchTask {
        std::vector<ActiveProcessor> active;
        size_t                       total_streams = 0;
        int                          propose_step  = 0;
        size_t                       vocab_size    = 0;
        torch::Tensor                draft_tokens;  // [B,P] or [B,P+1]
        // Async MTP: recorded on the producer stream after draft_tokens are
        // final. run() orders its D2H read after this event.
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
    };

    struct LaunchResult {
        torch::Tensor                         packed_allow_mask_gpu;   // CUDA-only [active_rows, ceil(V/32)] int32
        torch::Tensor                         logits_row_indices_gpu;  // CUDA-only [active_rows] int32
        bool                                  has_active_processor = false;
        std::vector<std::optional<ErrorInfo>> processor_errors;
        // Non-CUDA fallback consumes these directly.
        torch::Tensor packed_allow_mask_cpu_lifetime;
        torch::Tensor logits_row_indices_cpu_lifetime;
        torch::Tensor spec_cap_cpu;

        // Async MTP extensions. spec_cap_gpu mirrors spec_cap_cpu on device so
        // accept-len capping can stay tensorized without a host sync.
        // ready_event is recorded after the H2D uploads above; consumers must
        // block their stream on it. consumed_event must be recorded by the
        // consumer after its last GPU read; run() waits on it before reusing
        // the pinned scratch buffers (single-flight enforcement).
        torch::Tensor                 spec_cap_gpu;
        std::shared_ptr<torch::Event> ready_event;
        std::shared_ptr<torch::Event> consumed_event;
    };

    SpecLogitsVerifyRunner() = default;

    SpecLogitsVerifyRunner(const SpecLogitsVerifyRunner&)            = delete;
    SpecLogitsVerifyRunner& operator=(const SpecLogitsVerifyRunner&) = delete;

    LaunchResult run(const LaunchTask& task);
    static void  applyMaskToLogits(torch::Tensor& logits, const LaunchResult& result, size_t vocab_size);

private:
    struct VerifyShape {
        size_t batch_size       = 0;
        int    propose_step     = 0;
        size_t vocab_size       = 0;
        size_t bitmask_words    = 0;
        size_t compact_rows     = 0;
        size_t words_per_stream = 0;
    };

    struct ActiveStreamLayout {
        std::vector<size_t>  stream_indices;
        std::vector<int32_t> compact_slot_by_stream;
    };

    struct MergeProcessorMasksResult {
        std::vector<std::optional<ErrorInfo>> processor_errors;
    };

    ActiveStreamLayout buildActiveStreamLayout(const LaunchTask& task) const;
    void               ensureBuffersFit(const VerifyShape& shape);
    void               materializeDraftTokensToCpu(const LaunchTask& task);
    void               initializeCompactRows(const ActiveStreamLayout& layout, const VerifyShape& shape);
    MergeProcessorMasksResult
    mergeProcessorMasks(const LaunchTask& task, const ActiveStreamLayout& layout, const VerifyShape& shape);
    LaunchResult makeResult(const VerifyShape& shape);

    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;      // [active_rows, W] pinned int32; bit=1 allow
    torch::Tensor merged_bitmask_gpu_;      // [active_rows, W] device int32
    torch::Tensor logits_row_indices_cpu_;  // [active_rows] pinned int32
    torch::Tensor logits_row_indices_gpu_;  // [active_rows] device int32
    torch::Tensor spec_cap_cpu_;
#if USING_CUDA
    torch::Tensor                 spec_cap_gpu_;
    std::shared_ptr<torch::Event> last_consumed_event_;
#endif
};

}  // namespace rtp_llm
