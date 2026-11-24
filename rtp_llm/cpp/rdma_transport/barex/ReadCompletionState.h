#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"

namespace rtp_llm::rdma_transport::detail {

struct ReadCompletionState {
    struct ChunkResult {
        bool done{false};
        bool ok{false};
    };

    explicit ReadCompletionState(size_t chunk_count = 0): chunks(chunk_count) {}

    void completeChunk(size_t index, bool ok) {
        std::lock_guard<std::mutex> lock(mutex);
        if (chunks[index].done) {
            return;
        }
        chunks[index].done = true;
        chunks[index].ok   = ok;
        ++completed;
        if (completed == chunks.size()) {
            completion_confirmed->store(true, std::memory_order_release);
        }
        cv.notify_one();
    }

    bool rejectBeforeSubmission(size_t index, ::accl::barex::BarexResult result) {
        // Barex ReadBatch returns STAT/ARG from its preflight checks without
        // submitting NIC work or invoking its callback. Other errors are not
        // proof of quiescence and must still wait for a completion receipt.
        if (result != ::accl::barex::BAREX_ERR_STAT && result != ::accl::barex::BAREX_ERR_ARG) {
            return false;
        }
        completeChunk(index, false);
        return true;
    }

    std::mutex                         mutex;
    std::condition_variable            cv;
    std::vector<ChunkResult>           chunks;
    size_t                             completed{0};
    std::shared_ptr<std::atomic<bool>> completion_confirmed{std::make_shared<std::atomic<bool>>(false)};
};

}  // namespace rtp_llm::rdma_transport::detail
