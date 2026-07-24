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
// that reads its views, before calling run() again on the same runner. The runner
// separately waits for asynchronous H2D reads before mutating pinned CPU buffers.
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
    };

    struct LaunchResult {
        torch::Tensor                         packed_allow_mask_gpu;   // CUDA-only [active_rows, ceil(V/32)] int32
        torch::Tensor                         logits_row_indices_gpu;  // CUDA-only [active_rows] int32
        bool                                  has_active_processor = false;
        std::vector<std::optional<ErrorInfo>> processor_errors;
        // CUDA keeps these pinned H2D sources alive; non-CUDA fallback consumes
        // them directly and avoids an upload followed by an immediate readback.
        torch::Tensor packed_allow_mask_cpu_lifetime;
        torch::Tensor logits_row_indices_cpu_lifetime;
        torch::Tensor spec_cap_cpu;
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
    void         waitForPendingHostUploads();

    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;      // [active_rows, W] pinned int32; bit=1 allow
    torch::Tensor merged_bitmask_gpu_;      // [active_rows, W] device int32
    torch::Tensor logits_row_indices_cpu_;  // [active_rows] pinned int32
    torch::Tensor logits_row_indices_gpu_;  // [active_rows] device int32
    torch::Tensor spec_cap_cpu_;
    // Guards host mutation of merged_bitmask_cpu_ and logits_row_indices_cpu_;
    // correctness must not depend on a later sampler D2H stream synchronization.
    std::shared_ptr<torch::Event> pending_host_upload_;
};

}  // namespace rtp_llm
