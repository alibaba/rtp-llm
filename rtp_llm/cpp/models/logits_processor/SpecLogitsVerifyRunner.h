#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/SpecLogitsProcessorTypes.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

// Orchestration glue between MtpExecutor and per-stream SpecLogitsProcessors.
// Owns the reusable CPU/GPU bitmask + spec-cap scratch buffers and tracks which
// rows were filled in the previous round so the next call only has to reset and
// re-upload the changed rows (sparse-row H2D stays O(active streams), not
// O(B*(P+1)*W)). The packed-bitmask GPU output is consumed by
// MtpBatchStreamProcessor::gatherSpecSamplerInput.
class SpecLogitsVerifyRunner {
public:
    // Per-stream spec processor entry collected by the caller. Carrying the
    // stream id / lengths here lets callers stay free of GenerateStream coupling
    // inside the runner.
    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        size_t                 stream_idx    = 0;
        size_t                 processor_idx = 0;
        uint64_t               stream_id     = 0;
    };

    struct LaunchTask {
        std::vector<ActiveProcessor>  active;
        size_t                        total_streams = 0;
        int                           propose_step  = 0;
        size_t                        vocab_size    = 0;
        torch::Tensor                 draft_tokens;          // [B,P] or [B,P+1]
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
    };

    struct LaunchResult {
        torch::Tensor                      spec_vocab_mask_gpu;
        torch::Tensor                      spec_cap_gpu;
        bool                               has_active_processor = false;
        std::vector<SpecLogitsProcessorId> applied_processors;
        torch::Tensor                      spec_vocab_mask_cpu_owner;
        torch::Tensor                      spec_cap_cpu_owner;
    };

    SpecLogitsVerifyRunner() = default;

    SpecLogitsVerifyRunner(const SpecLogitsVerifyRunner&)            = delete;
    SpecLogitsVerifyRunner& operator=(const SpecLogitsVerifyRunner&) = delete;

    // Drives every eligible SpecLogitsProcessor on the given streams, ANDs their
    // packed bitmasks into the merged buffer, and returns GPU views ready for
    // sampler consumption. Returns an empty result when no processor was
    // applied; in that case any rows touched by the previous call are reset to
    // allow-all on the GPU side.
    LaunchResult buildInline(const LaunchTask& task);

private:
    void ensureBuffersFit(size_t total_streams, int propose_step, size_t vocab_size, size_t bitmask_words);
    void materializeDraftTokensToCpu(const LaunchTask& task);

    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;
    torch::Tensor merged_bitmask_gpu_;
    torch::Tensor spec_cap_cpu_;
    torch::Tensor spec_cap_gpu_;
    // Rows last filled by an active grammar processor; lets the next call only reset+upload
    // those rows instead of the full B*(P+1)*W buffer.
    std::vector<size_t> last_active_stream_rows_;
};

}  // namespace rtp_llm
