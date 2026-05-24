#pragma once

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/models/logits_processor/SpecLogitsProcessor.h"

#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rtp_llm {

// Backend-agnostic helper that owns one private CUDA stream and a set of
// reusable host/device buffers, and lets the executor turn a per-stream draft
// token batch into (a) a per-row grammar bitmask block on GPU and (b) a
// per-stream grammar accept-len cap on GPU.
//
// Design notes (single-sequence; multi-seq deferred per plan §九.2):
//   - The helper does **not** spawn a CPU worker thread. All CPU matcher
//     walks run on the executor's main thread, on which target verify
//     forward has already been launched as a non-blocking GPU op. The CPU
//     walk overlaps naturally with the GPU forward.
//   - Draft tokens are pulled from GPU via an async D2H on the private CUDA
//     stream; the helper then host-syncs on that copy before walking (see the
//     xgrammar-hot-path-allow line in launch()). The host sync does NOT
//     block the main CUDA stream's forward kernel.
//   - Once the bitmask + cap are filled on host they are pushed back via an
//     async H2D on the private stream, and a final ready_event is recorded.
//     The main stream waits this event right before applying the mask kernel.
class SpecGrammarVerifyHelper {
public:
    SpecGrammarVerifyHelper();
    ~SpecGrammarVerifyHelper();

    SpecGrammarVerifyHelper(const SpecGrammarVerifyHelper&)            = delete;
    SpecGrammarVerifyHelper& operator=(const SpecGrammarVerifyHelper&) = delete;

    struct LaunchTask {
        // (processor, row slot in the score-batch). One entry per
        // grammar-active stream; non-active streams stay out of this list and
        // get cap=propose_step + allow-all rows automatically.
        std::vector<SpecLogitsProcessorPtr> active;
        std::vector<size_t>                 active_row_slots;

        // Total number of streams contributing to the score-batch. cap_gpu is
        // sized [total_streams]; bitmask_gpu is sized [total_streams * (P+1), W].
        size_t total_streams = 0;

        // Either provide draft_tokens_gpu + draft_tokens_ready_event for an
        // async D2H, or draft_tokens_cpu_fastpath for a memcpy-only short path
        // (P==1 today, since the executor already has a CPU mirror).
        torch::Tensor draft_tokens_gpu;       // [total_streams, propose_step] CUDA int32
        torch::Event* draft_tokens_ready_event = nullptr;
        torch::Tensor draft_tokens_cpu_fastpath;  // [total_streams, propose_step] CPU int32

        int    propose_step = 0;
        size_t vocab_size   = 0;
    };

    struct LaunchResult {
        // Empty when there is nothing to do (no active streams or P==0).
        torch::Tensor                  bitmask_gpu;     // [total_streams*(P+1), W] int32
        torch::Tensor                  cap_gpu;         // [total_streams] int32
        std::shared_ptr<torch::Event>  ready_event;     // recorded on xg_stream after H2D
        // Caller MUST record this event on the main stream once it is done
        // reading bitmask_gpu/cap_gpu (e.g. right after sampler->forward returns).
        // The next launch() waits on this event on its private xg_stream before
        // issuing the next H2D, preventing the helper from clobbering the buffer
        // while a previous step's mask kernel could still be reading it.
        std::shared_ptr<torch::Event>  consumer_done_event;
    };

    // Synchronously executes steps 1-7 listed in the class doc and returns
    // tensors carved out of the helper's reusable buffers (caller must NOT
    // hold the result tensors past the next launch()).
    LaunchResult launch(const LaunchTask& task);

private:
    void ensureBuffersFit_(size_t total_streams, int propose_step, size_t bitmask_words);
    LaunchResult makeEmptyResult_();

    cuda_graph::GraphStream xg_stream_;  // private CUDA stream, high priority
    // Set by the previous launch()'s LaunchResult.consumer_done_event when the
    // caller records it on the main stream. We wait on this event on
    // xg_stream_ at the start of the next launch() so subsequent H2D into the
    // reusable bitmask_gpu_/cap_gpu_ buffers cannot overlap with the previous
    // step's mask kernel still reading from them on the main stream.
    std::shared_ptr<torch::Event> last_consumer_done_event_;
    // Reusable buffers (resized on demand).
    torch::Tensor drafts_pinned_;   // [B_cap, P_cap] CPU pinned int32
    torch::Tensor bitmask_cpu_;     // [B_cap*(P_cap+1), W_cap] CPU pinned int32
    torch::Tensor bitmask_gpu_;     // [B_cap*(P_cap+1), W_cap] CUDA int32
    torch::Tensor cap_cpu_;         // [B_cap] CPU pinned int32
    torch::Tensor cap_gpu_;         // [B_cap] CUDA int32

    size_t buf_streams_ = 0;
    int    buf_propose_ = 0;
    size_t buf_words_   = 0;
};

}  // namespace rtp_llm
