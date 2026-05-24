#include "rtp_llm/cpp/models/logits_processor/SpecGrammarVerifyHelper.h"

#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cstring>
#include <utility>

namespace rtp_llm {

SpecGrammarVerifyHelper::SpecGrammarVerifyHelper():
    xg_stream_(cuda_graph::graphGetStreamFromPool(/*is_high_priority=*/true)) {}

namespace {
// Build a torch::Stream view of a GraphStream so torch::Event::block / record
// can be called on it.
inline torch::Stream toTorchStream(const cuda_graph::GraphStream& s) {
    return torch::Stream(s);
}
}  // namespace

SpecGrammarVerifyHelper::~SpecGrammarVerifyHelper() = default;

void SpecGrammarVerifyHelper::ensureBuffersFit_(size_t total_streams,
                                                int    propose_step,
                                                size_t bitmask_words) {
    const bool need_resize = total_streams > buf_streams_ || propose_step > buf_propose_
                             || bitmask_words > buf_words_;
    if (!need_resize) {
        return;
    }
    const size_t B = std::max(total_streams, buf_streams_);
    const int    P = std::max(propose_step, buf_propose_);
    const size_t W = std::max(bitmask_words, buf_words_);

    auto cpu_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU).pinned_memory(true);
    auto gpu_int32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);

    drafts_pinned_ = torch::empty({static_cast<int64_t>(B), static_cast<int64_t>(P > 0 ? P : 1)},
                                  cpu_int32);

    const int64_t M = static_cast<int64_t>(B) * static_cast<int64_t>(P + 1);
    bitmask_cpu_ = torch::empty({M, static_cast<int64_t>(W)}, cpu_int32);
    bitmask_gpu_ = torch::empty({M, static_cast<int64_t>(W)}, gpu_int32);

    cap_cpu_ = torch::empty({static_cast<int64_t>(B)}, cpu_int32);
    cap_gpu_ = torch::empty({static_cast<int64_t>(B)}, gpu_int32);

    buf_streams_ = B;
    buf_propose_ = P;
    buf_words_   = W;
}

SpecGrammarVerifyHelper::LaunchResult SpecGrammarVerifyHelper::makeEmptyResult_() {
    return LaunchResult{};
}

SpecGrammarVerifyHelper::LaunchResult
SpecGrammarVerifyHelper::launch(const LaunchTask& task) {
    if (task.total_streams == 0 || task.propose_step <= 0 || task.vocab_size == 0) {
        return makeEmptyResult_();
    }

    const int    P = task.propose_step;
    const size_t B = task.total_streams;
    const size_t W = SpecLogitsProcessor::bitmaskWordCount(task.vocab_size);
    const size_t M = B * static_cast<size_t>(P + 1);

    ensureBuffersFit_(B, P, W);

    // ── 0. Defensive ordering: wait for the previous step's main-stream
    //       consumer (mask kernel) to finish reading the reusable
    //       bitmask_gpu_/cap_gpu_ buffers before we overwrite them on
    //       xg_stream_. This is a GPU-side stream wait — the main thread is
    //       not blocked. On the very first launch, last_consumer_done_event_
    //       is null and we skip.
    if (last_consumer_done_event_) {
        last_consumer_done_event_->block(toTorchStream(xg_stream_));
        last_consumer_done_event_.reset();
    }

    // ── 1. Get draft tokens onto pinned host memory ───────────────────────────
    int32_t* drafts_host = drafts_pinned_.data_ptr<int32_t>();
    if (task.draft_tokens_cpu_fastpath.defined()) {
        // Fast path: caller already has a host mirror (P==1 today). Skip D2H.
        RTP_LLM_CHECK_WITH_INFO(
            static_cast<size_t>(task.draft_tokens_cpu_fastpath.numel()) >= B * static_cast<size_t>(P),
            "spec_grammar_helper fastpath cpu mirror too small: numel=%ld, expected>=%lu",
            (long)task.draft_tokens_cpu_fastpath.numel(),
            B * static_cast<size_t>(P));
        std::memcpy(drafts_host,
                    task.draft_tokens_cpu_fastpath.data_ptr<int32_t>(),
                    B * static_cast<size_t>(P) * sizeof(int32_t));
    } else if (task.draft_tokens_gpu.defined()) {
        if (task.draft_tokens_ready_event) {
            task.draft_tokens_ready_event->block(toTorchStream(xg_stream_));
        }
        // The helper performs an async D2H of draft tokens on its private
        // stream and host-waits the completion event. This host wait is by
        // design (the CPU walk needs the draft tokens) and does NOT block the
        // main CUDA stream's target verify forward kernel; the CPU walk that
        // follows runs concurrently with that in-flight GPU work.
        auto draft_view  = task.draft_tokens_gpu.view({-1, P}).slice(0, 0, B);
        auto pinned_view = drafts_pinned_.slice(0, 0, B);
        {
            cuda_graph::GraphStreamGuard guard(xg_stream_);
            pinned_view.copy_(draft_view, /*non_blocking=*/true);
        }
        torch::Event d2h_done = cuda_graph::makeGraphEvent();
        d2h_done.record(toTorchStream(xg_stream_));
        d2h_done.synchronize();  // xgrammar-hot-path-allow: intentional host wait for our private xg_stream D2H; main stream forward keeps running
    } else {
        return makeEmptyResult_();
    }

    // ── 2. Initialise bitmask = allow-all and cap = propose_step ──────────────
    int32_t* bitmask_host = bitmask_cpu_.data_ptr<int32_t>();
    int32_t* cap_host     = cap_cpu_.data_ptr<int32_t>();
    std::memset(bitmask_host,
                SpecLogitsProcessor::kBitmaskAllowAll,
                M * W * sizeof(int32_t));
    std::fill_n(cap_host, B, static_cast<int32_t>(P));

    // ── 3. CPU matcher walk for each grammar-active stream ────────────────────
    // Per-row try/catch: if a single backend's tryAcceptAndFillBitmask throws
    // (e.g. xgrammar matcher corruption, FillNextTokenBitmask OOM), restore
    // that row's block to allow-all and cap=P so the rest of the batch can
    // proceed with safe-permissive defaults instead of aborting the whole
    // decode step.
    for (size_t k = 0; k < task.active.size(); ++k) {
        const auto&  proc = task.active[k];
        const size_t row  = task.active_row_slots[k];
        if (!proc || row >= B) {
            continue;
        }
        if (!proc->isSpecVerifyEligible()) {
            continue;  // leave allow-all + cap=P
        }
        int32_t*       bitmask_block = bitmask_host + row * static_cast<size_t>(P + 1) * W;
        const int32_t* draft_row     = drafts_host + row * static_cast<size_t>(P);

        try {
            const int cap = proc->tryAcceptAndFillBitmask(draft_row, P, bitmask_block, W);
            cap_host[row] = static_cast<int32_t>(cap);
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING(
                "spec grammar verify row %zu: tryAcceptAndFillBitmask threw (%s); "
                "degrading row to allow-all + cap=P, batch continues.",
                row, e.what());
            std::memset(bitmask_block,
                        SpecLogitsProcessor::kBitmaskAllowAll,
                        static_cast<size_t>(P + 1) * W * sizeof(int32_t));
            cap_host[row] = static_cast<int32_t>(P);
        } catch (...) {
            RTP_LLM_LOG_WARNING(
                "spec grammar verify row %zu: tryAcceptAndFillBitmask threw unknown; "
                "degrading row to allow-all + cap=P, batch continues.",
                row);
            std::memset(bitmask_block,
                        SpecLogitsProcessor::kBitmaskAllowAll,
                        static_cast<size_t>(P + 1) * W * sizeof(int32_t));
            cap_host[row] = static_cast<int32_t>(P);
        }
    }

    // ── 4. Async H2D bitmask + cap on the private stream, record ready_event ──
    auto ready_event      = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    auto consumer_done    = std::make_shared<torch::Event>(cuda_graph::makeGraphEvent());
    auto bitmask_gpu_view = bitmask_gpu_.slice(0, 0, static_cast<int64_t>(M));
    auto cap_gpu_view     = cap_gpu_.slice(0, 0, static_cast<int64_t>(B));
    {
        cuda_graph::GraphStreamGuard guard(xg_stream_);
        bitmask_gpu_view.copy_(bitmask_cpu_.slice(0, 0, static_cast<int64_t>(M)),
                                /*non_blocking=*/true);
        cap_gpu_view.copy_(cap_cpu_.slice(0, 0, static_cast<int64_t>(B)), /*non_blocking=*/true);
    }
    ready_event->record(toTorchStream(xg_stream_));

    // Stash the consumer_done event so step (0) of the NEXT launch can wait
    // it on xg_stream_ before issuing the next H2D into the same buffers.
    last_consumer_done_event_ = consumer_done;

    return LaunchResult{
        std::move(bitmask_gpu_view),
        std::move(cap_gpu_view),
        std::move(ready_event),
        std::move(consumer_done),
    };
}

}  // namespace rtp_llm
