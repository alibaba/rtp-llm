#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>

namespace rtp_llm::block_transfer_engine_test {
// Link wrappers affect only an explicitly armed test operation. CUDA work is
// real; a host function holds its stream before the first DMA executes.
struct RemoteCopyFailure {
    std::mutex              mutex;
    std::condition_variable cv;
    int                     fail_registration{0};
    bool                    fail_second_copy{false};
    std::atomic<int>        registrations{0}, copies{0}, early_frees{0};
    bool                    queued{false}, injected{false}, opened{false};
    std::vector<void*>      deferred_frees;
    void                    open();
    bool                    waitQueued();
    bool                    waitInjected();
};
std::shared_ptr<RemoteCopyFailure> armRemoteCopyFailure(int registration, bool second_copy = false);
void                               retireRemoteCopyFailure(const std::shared_ptr<RemoteCopyFailure>&);
}  // namespace rtp_llm::block_transfer_engine_test
