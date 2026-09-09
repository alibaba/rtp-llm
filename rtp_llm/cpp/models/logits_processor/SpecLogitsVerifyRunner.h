#pragma once

#include <atomic>
#include <memory>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

class SpecLogitsVerifyRunner {
    struct DraftTokenSlot {
        torch::Tensor storage;
        // Set only after the worker's existing D2H wait has succeeded.
        std::atomic<bool> copy_completed{false};
    };

public:
    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        size_t                 stream_idx      = 0;
        size_t                 processor_idx   = 0;
        uint64_t               stream_id       = 0;
        int64_t                base_seq_len    = 0;
        int64_t                base_output_len = 0;
    };

    struct DraftTokensTransfer {
        // Per-launch ownership: no worker or later decode round may overwrite
        // this pinned destination while the asynchronous D2H is in flight.
        torch::Tensor                 cpu_tokens;     // contiguous int32 [B, P]
        torch::Tensor                 source_tokens;  // original producer storage
        torch::Tensor                 packed_tokens;  // contiguous int32 D2H source
        std::shared_ptr<torch::Event> ready_event;    // absent for synchronous CPU input
        size_t                       total_streams = 0;
        int                          propose_step  = 0;

    private:
        friend class SpecLogitsVerifyRunner;
        // The lease lasts until all owners of this transfer have released it.
        std::shared_ptr<DraftTokenSlot> slot;
    };

    struct LaunchTask {
        std::vector<ActiveProcessor> active;
        size_t                       total_streams = 0;
        int                          propose_step  = 0;
        size_t                       vocab_size    = 0;

        // Shape [B, P], dtype int32/int64, CPU or CUDA.
        torch::Tensor                 draft_tokens;
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
        std::shared_ptr<DraftTokensTransfer> draft_transfer;
    };

    struct LaunchResult {
        // Bool vocab mask, shape [B * (P + 1), vocab_size], CUDA when active.
        // true means masked. This keeps the generic sampler path independent
        // from grammar-specific packed bitmask kernels.
        torch::Tensor                      spec_vocab_mask_gpu;
        torch::Tensor                      spec_cap_gpu;  // [B] int32 CUDA
        std::shared_ptr<torch::Event>      ready_event;
        std::shared_ptr<torch::Event>      consumed_event;
        bool                               has_active_processor = false;
        // Processors skipped because isSpecVerifyEligible() was false at build
        // time (e.g. grammar already reported an error on that stream). Their
        // mask rows stay all-allow and their cap stays at propose_step.
        size_t                             skipped_ineligible_processors = 0;
        std::vector<SpecLogitsProcessorId> applied_processors;

        // Keep H2D sources alive until ready_event has completed. The runner
        // reuses scratch buffers across decode rounds, while these tensors own
        // the actual async transfer source for this result.
        torch::Tensor spec_vocab_mask_cpu_owner;
        torch::Tensor spec_cap_cpu_owner;
    };

    SpecLogitsVerifyRunner();

    // Caller-side submission only: does not inspect processors or wait for GPU
    // completion. The worker later consumes the owned transfer in buildInline.
    void enqueueDraftTokensToCpu(LaunchTask& task);
    LaunchResult buildInline(const LaunchTask& task);

private:
    std::shared_ptr<DraftTokenSlot> acquireDraftTokenSlot(int64_t elements);
    void ensureBuffersFit(size_t total_streams,
                          size_t active_streams,
                          int    propose_step,
                          size_t vocab_size,
                          size_t bitmask_words);
    void materializeDraftTokensToCpu(const LaunchTask& task);
    void unpackMergedBitmaskToVocabMask(const torch::Tensor& mask_cpu,
                                        size_t               rows,
                                        size_t               vocab_size,
                                        size_t               bitmask_words);

private:
    struct CpuArtifactSlot {
        torch::Tensor                 mask;
        torch::Tensor                 cap;
        std::shared_ptr<torch::Event> ready_event;
    };

    torch::Stream copy_stream_;
    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;
    torch::Tensor spec_cap_cpu_;
    std::vector<CpuArtifactSlot> cpu_artifact_slots_;
    // Accessed only by enqueue's producer thread, never by the worker.
    // Two cached slots cover producer/consumer overlap; additional live
    // transfers fall back to independent allocations instead of growing a pool.
    std::vector<std::shared_ptr<DraftTokenSlot>> draft_token_slots_;
};

}  // namespace rtp_llm
