#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_set>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

namespace rtp_llm {

class BaseLogitsProcessor;

// Glue between MtpExecutor and per-stream SpecLogitsProcessors; emits a GPU bool disallow mask.
class SpecLogitsVerifyRunner {
public:
    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        // BaseLogitsProcessor side of the same object (multi-inheritance offset);
        // used as identity key in SamplerInputs::spec_applied_processors.
        BaseLogitsProcessor* base_id    = nullptr;
        size_t               stream_idx = 0;
    };

    struct LaunchTask {
        std::vector<ActiveProcessor>  active;
        size_t                        total_streams = 0;
        int                           propose_step  = 0;
        size_t                        vocab_size    = 0;
        torch::Tensor                 draft_tokens;  // [B,P] or [B,P+1]
        std::shared_ptr<torch::Event> draft_tokens_ready_event;
    };

    struct LaunchResult {
        torch::Tensor                            spec_vocab_mask_gpu;  // [rows, V] bool, true=disallow
        bool                                     has_active_processor = false;
        std::unordered_set<BaseLogitsProcessor*> applied_processors;
        torch::Tensor                            spec_vocab_mask_cpu_owner;
        torch::Tensor                            spec_cap_cpu_owner;
    };

    SpecLogitsVerifyRunner() = default;

    SpecLogitsVerifyRunner(const SpecLogitsVerifyRunner&)            = delete;
    SpecLogitsVerifyRunner& operator=(const SpecLogitsVerifyRunner&) = delete;

    LaunchResult buildInline(const LaunchTask& task);
    static void
    applyMaskToLogits(const torch::Tensor& logits, const torch::Tensor& spec_vocab_mask_gpu, size_t vocab_size);

private:
    void ensureBuffersFit(size_t total_streams, int propose_step, size_t vocab_size, size_t bitmask_words);
    void materializeDraftTokensToCpu(const LaunchTask& task);
    void unpackRowToBoolDisallow(size_t row, size_t vocab_size, size_t bitmask_words);

    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;  // [rows, W] int32 packed; bit=1 allow
    torch::Tensor disallow_mask_cpu_;   // [rows, V] pinned bool; true=disallow
    torch::Tensor disallow_mask_gpu_;   // [rows, V] cuda bool; mirrors disallow_mask_cpu_
    torch::Tensor spec_cap_cpu_;
    // Rows last filled by an active grammar processor; lets the next call only
    // reset + re-unpack + re-upload those rows instead of the full B*(P+1) buffer.
    std::vector<size_t> last_active_stream_rows_;
};

}  // namespace rtp_llm
