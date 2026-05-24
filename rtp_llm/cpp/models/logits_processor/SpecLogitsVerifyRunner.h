#pragma once

#include <memory>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

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

        // Shape [B, P], dtype int32/int64, CPU or CUDA.
        torch::Tensor                 draft_tokens;
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
    };

    struct LaunchResult {
        // Bool vocab mask, shape [B * (P + 1), vocab_size], CUDA when active.
        // true means masked. This keeps the generic sampler path independent
        // from grammar-specific packed bitmask kernels.
        torch::Tensor                 spec_vocab_mask_gpu;
        torch::Tensor                 spec_cap_gpu;  // [B] int32 CUDA
        std::shared_ptr<torch::Event> ready_event;
        bool                          has_active_processor = false;
    };

    SpecLogitsVerifyRunner();

    LaunchResult buildInline(const LaunchTask& task);

private:
    void ensureBuffersFit(size_t total_streams, int propose_step, size_t vocab_size, size_t bitmask_words);
    void materializeDraftTokensToCpu(const LaunchTask& task);
    void unpackMergedBitmaskToVocabMask(size_t rows, size_t vocab_size, size_t bitmask_words);

private:
    torch::Stream copy_stream_;
    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;
    torch::Tensor spec_vocab_mask_cpu_;
    torch::Tensor spec_vocab_mask_gpu_;
    torch::Tensor spec_cap_cpu_;
    torch::Tensor spec_cap_gpu_;
};

}  // namespace rtp_llm
