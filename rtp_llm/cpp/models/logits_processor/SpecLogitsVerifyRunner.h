#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

// Builds compact packed allow-masks for MTP target verification. The runner is
// single-flight: result tensors are views into reusable scratch buffers, and
// the next build waits for the consumer event before reusing them.
class SpecLogitsVerifyRunner {
public:
    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        size_t                 stream_idx      = 0;
        size_t                 processor_idx   = 0;
        uint64_t               stream_id       = 0;
        int64_t                base_seq_len    = 0;
        int64_t                base_output_len = 0;
    };

    struct LaunchTask {
        std::vector<ActiveProcessor> active;
        size_t                       total_streams = 0;
        int                          propose_step  = 0;
        size_t                       vocab_size    = 0;
        torch::Tensor                draft_tokens;  // [B, P] or [B, P + 1]
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
    };

    struct LaunchResult {
        // Each int32 word covers 32 vocabulary entries (bit=1 means allowed).
        torch::Tensor packed_allow_mask_gpu;
        torch::Tensor logits_row_indices_gpu;
        torch::Tensor spec_cap_gpu;  // [B] int32 CUDA

        std::shared_ptr<torch::Event>      ready_event;
        std::shared_ptr<torch::Event>      consumed_event;
        bool                               has_active_processor = false;
        std::vector<SpecLogitsProcessorId> applied_processors;

        torch::Tensor packed_allow_mask_cpu_lifetime;
        torch::Tensor logits_row_indices_cpu_lifetime;
        torch::Tensor spec_cap_cpu_lifetime;
    };

    SpecLogitsVerifyRunner();

    LaunchResult buildInline(const LaunchTask& task);
    static void applyMaskToLogits(const torch::Tensor& logits, const LaunchResult& result, size_t vocab_size);

private:
    void ensureBuffersFit(size_t total_streams, int propose_step, size_t bitmask_words, size_t compact_rows);
    void materializeDraftTokensToCpu(const LaunchTask& task);

private:
    torch::Stream copy_stream_;
    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;
    torch::Tensor merged_bitmask_gpu_;
    torch::Tensor logits_row_indices_cpu_;
    torch::Tensor logits_row_indices_gpu_;
    torch::Tensor spec_cap_cpu_;
    torch::Tensor spec_cap_gpu_;
    std::shared_ptr<torch::Event> last_consumed_event_;
};

}  // namespace rtp_llm
