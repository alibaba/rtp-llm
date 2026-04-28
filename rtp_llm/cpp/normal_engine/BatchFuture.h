#pragma once

#include <atomic>
#include <list>
#include <memory>
#include "absl/status/status.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

// Per-batch async handoff between the launch thread and the bookkeeping
// thread. The result thread owns post-GPU dispatch/specUpdate/housekeeping and
// marks bookkeeping_done before the next scheduler iteration reads stream state.
struct BatchFuture {
    // Streams that participated in this batch. Owned references are kept
    // alive until bookkeeping completes so per-stream specUpdate / KV
    // release does not race with main-thread enqueue.
    std::list<GenerateStreamPtr> streams;

    // launchTimeUs records when the main thread submitted the GPU work.
    // Used by metrics + as a coarse staleness guard.
    int64_t launch_time_us = 0;

    // bookkeeping_done flips to true once the result thread (or, on the
    // synchronous fallback path, the main thread itself) finishes
    // dispatchDecode + per-stream updates. asyncStep waits on this before
    // re-entering scheduler so seq_len etc. stay accurate.
    std::atomic<bool> bookkeeping_done{false};

    // Status reported by the bookkeeping pass. asyncStep surfaces it on
    // the next iteration so error handling matches the sync path.
    absl::Status bookkeeping_status = absl::OkStatus();
};

using BatchFuturePtr = std::shared_ptr<BatchFuture>;

}  // namespace rtp_llm
