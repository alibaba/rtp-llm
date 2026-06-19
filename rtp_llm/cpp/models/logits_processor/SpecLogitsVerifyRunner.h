#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "rtp_llm/cpp/models/SpecLogitsProcessorTypes.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"

namespace rtp_llm {

// Orchestration glue between MtpExecutor and per-stream SpecLogitsProcessors.
// Owns the reusable CPU-side bitmask + spec-cap scratch buffers and tracks
// which rows were filled in the previous round so the next call only resets
// and re-uploads the changed rows. The GPU-side artifact handed to the sampler
// is a bool disallow mask (true = disallow); the GPU bitmask is unpacked from
// the CPU packed bitmask once per build, so consumers can apply it with a
// single masked_fill_ on every platform (no per-arch kernel branch).
class SpecLogitsVerifyRunner {
public:
    using ErrorSink = std::function<void(ErrorCode, const std::string&)>;

    struct ActiveProcessor {
        SpecLogitsProcessorPtr processor;
        size_t                 stream_idx    = 0;
        size_t                 processor_idx = 0;
        uint64_t               stream_id     = 0;
        ErrorSink              error_sink;
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
        torch::Tensor                      spec_vocab_mask_gpu;  // [rows, V] bool, true=disallow
        torch::Tensor                      spec_cap_gpu;
        bool                               has_active_processor = false;
        std::vector<SpecLogitsProcessorId> applied_processors;
        torch::Tensor                      spec_vocab_mask_cpu_owner;
        torch::Tensor                      spec_cap_cpu_owner;
    };

    SpecLogitsVerifyRunner() = default;

    SpecLogitsVerifyRunner(const SpecLogitsVerifyRunner&)            = delete;
    SpecLogitsVerifyRunner& operator=(const SpecLogitsVerifyRunner&) = delete;

    LaunchResult buildInline(const LaunchTask& task);

private:
    void ensureBuffersFit(size_t total_streams, int propose_step, size_t vocab_size, size_t bitmask_words);
    void materializeDraftTokensToCpu(const LaunchTask& task);
    void unpackRowToBoolDisallow(size_t row, size_t vocab_size, size_t bitmask_words);

    torch::Tensor draft_tokens_cpu_;
    torch::Tensor processor_bitmask_cpu_;
    torch::Tensor merged_bitmask_cpu_;        // [rows, W] int32 packed; bit=1 allow
    torch::Tensor disallow_mask_cpu_;         // [rows, V] pinned bool; true=disallow
    torch::Tensor disallow_mask_gpu_;         // [rows, V] cuda bool; mirrors disallow_mask_cpu_
    torch::Tensor spec_cap_cpu_;
    torch::Tensor spec_cap_gpu_;
    // Rows last filled by an active grammar processor; lets the next call only
    // reset + re-unpack + re-upload those rows instead of the full B*(P+1) buffer.
    std::vector<size_t> last_active_stream_rows_;
};

}  // namespace rtp_llm
