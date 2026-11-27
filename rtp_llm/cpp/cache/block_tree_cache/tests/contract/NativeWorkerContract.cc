#include "rtp_llm/cpp/cache/block_tree_cache/ScopeRollback.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/PerRankBlockTransferEngine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
using namespace rtp_llm;
using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

TransferResult safeResult(TransferStatus status = TransferStatus::OK) {
    TransferResult result(status);
    result.buffer_safe = true;
    return result;
}

class RecordedIo {
public:
    PerRankWorkerIO worker() {
        return {
            [](const TransferTask& batch) { return batch.descriptors().front().group_set_id <= 1; },
            [this](const TransferTask& batch, TransferAsyncContext::DoneCallback done) {
                std::unique_lock<std::mutex> lock(mutex_);
                if (draining_) {
                    lock.unlock();
                    done(safeResult());
                    return;
                }
                std::vector<size_t> positions;
                for (const auto& desc : batch.descriptors())
                    positions.push_back(desc.path_index);
                batches_.push_back({std::move(positions), std::move(done)});
                ready_.notify_all();
            },
        };
    }

    std::vector<std::vector<size_t>> awaitBatches(size_t count) {
        std::unique_lock<std::mutex> lock(mutex_);
        require(ready_.wait_for(lock, 2s, [&] { return batches_.size() >= count; }), "expected native owner batches");
        std::vector<std::vector<size_t>> positions;
        for (const auto& batch : batches_)
            positions.push_back(batch.positions);
        std::sort(positions.begin(), positions.end());
        return positions;
    }

    void complete(size_t first, TransferResult result) {
        TransferAsyncContext::DoneCallback done;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto batch = std::find_if(batches_.begin(), batches_.end(),
                                      [first](const auto& value) { return value.positions.front() == first; });
            require(batch != batches_.end() && batch->done, "callback must be owned by one recorded batch");
            done = std::move(batch->done);
        }
        done(std::move(result));
    }

    void drain() {
        std::vector<Batch> batches;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            draining_ = true;
            batches = std::move(batches_);
        }
        for (auto& batch : batches) {
            if (batch.done)
                batch.done(safeResult());
        }
    }

private:
    struct Batch {
        std::vector<size_t> positions;
        TransferAsyncContext::DoneCallback done;
    };
    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<Batch> batches_;
    bool draining_{false};
};

TransferDescriptor copy(size_t group, size_t position, bool load = true) {
    auto desc = load ? TransferDescriptor::hostToDevice(group, 1, {2}) :
                       TransferDescriptor::deviceToHost(group, {2}, 1);
    desc.path_index = position;
    return desc;
}

void boundedBatchesKeepResultOrder() {
    RecordedIo io;
    PerRankBlockTransferEngine engine({}, PerRankWorkerOptions{2, 2, 8, 0}, io.worker());
    block_tree_cache_detail::ScopeRollback drain([&] { io.drain(); });
    auto context = engine.execute(TransferTask(
        {copy(0, 0), copy(0, 1), copy(0, 2), copy(1, 3), copy(1, 4), copy(0, 5, false)}, 2s));
    require(io.awaitBatches(4) == std::vector<std::vector<size_t>>{{0, 1}, {2}, {3, 4}, {5}},
            "shared native owner must preserve groups, direction and the two-row limit");
    io.complete(5, safeResult());
    io.complete(2, safeResult(TransferStatus::DISK_IO_ERROR));
    io.complete(3, safeResult());
    require(!context->done(), "logical result must wait for the last physical batch");
    io.complete(0, safeResult());
    context->waitDone();
    auto result = context->result();
    require(!result.ok() && result.buffer_safe && result.block_errors ==
                std::vector<CacheCopyError>{CacheCopyError::NONE, CacheCopyError::NONE, CacheCopyError::IO_FAILED,
                                            CacheCopyError::NONE, CacheCopyError::NONE, CacheCopyError::NONE},
            "out-of-order completion must preserve result indexes and physical safety");
}

void admissionRetainsCapacityUntilCompletion() {
    RecordedIo io;
    PerRankBlockTransferEngine engine({}, PerRankWorkerOptions{1, 1, 1, 0}, io.worker());
    block_tree_cache_detail::ScopeRollback drain([&] { io.drain(); });
    auto load = engine.execute(TransferTask({copy(0, 0)}, 2s));
    require(io.awaitBatches(1) == std::vector<std::vector<size_t>>{{0}}, "first load must reach native owner");
    auto rejected = engine.execute(TransferTask({copy(0, 1)}, 2s));
    require(rejected->done() && !rejected->success() && rejected->result().buffer_safe,
            "occupied native load lane must reject without physical submission");
    auto evict = engine.execute(TransferTask({copy(0, 2, false)}, 2s));
    require(io.awaitBatches(2) == std::vector<std::vector<size_t>>{{0}, {2}},
            "one occupied direction must not block the other");
    const auto outstanding = engine.queueSizes();
    require(outstanding.load == 1 && outstanding.background == 1, "both accepted batches remain resident");
    engine.stopAdmission();
    auto stopped = engine.execute(TransferTask({copy(0, 3)}, 2s));
    require(stopped->done() && !stopped->success() && stopped->result().buffer_safe,
            "closing admission must reject before submission");
    io.complete(2, safeResult());
    io.complete(0, safeResult());
    engine.shutdown();
    require(load->done() && load->success() && evict->done() && evict->success(),
            "shutdown must drain accepted work and its callbacks");
}

void expiredWorkDoesNotReachIo() {
    RecordedIo io;
    PerRankBlockTransferEngine engine({}, PerRankWorkerOptions{1, 1, 1, 0}, io.worker());
    block_tree_cache_detail::ScopeRollback drain([&] { io.drain(); });
    auto stopped = engine.execute(TransferTask({copy(0, 0)}, 0ms));
    require(stopped->done() && stopped->errorInfo().code() == ErrorCode::DEADLINE_EXCEEDED
                && stopped->result().buffer_safe && engine.queueSizes().load == 0,
            "expired work must settle safely before admission");
}

void queuedDeadlineDoesNotRevokeSubmittedIo() {
    std::promise<void> submitted;
    auto started = submitted.get_future();
    std::promise<void> finish;
    auto finish_signal = finish.get_future();
    std::atomic<size_t> submitted_batches{0};
    PerRankWorkerIO io{
        [](const TransferTask&) { return true; },
        [&](const TransferTask&, TransferAsyncContext::DoneCallback done) {
            if (submitted_batches.fetch_add(1, std::memory_order_relaxed) == 0) {
                submitted.set_value();
                finish_signal.wait();
            }
            done(safeResult());
        },
    };
    PerRankBlockTransferEngine engine({}, PerRankWorkerOptions{1, 1, 2, 0}, std::move(io));
    block_tree_cache_detail::ScopeRollback unblock([&] { finish.set_value(); });
    TransferTask first({copy(0, 0)}, 250ms);
    auto active = engine.execute(first);
    require(started.wait_for(2s) == std::future_status::ready, "physical admission must reach the first batch");
    auto queued = engine.execute(TransferTask({copy(0, 1)}, 150ms));
    require(engine.queueSizes().load == 2 && !queued->done(), "second batch must occupy the native queue");
    std::this_thread::sleep_until(first.deadline() + 150ms);
    require(!active->done(), "accepted I/O stays resident until its producer completes");
    unblock.run();
    engine.shutdown();
    require(active->done() && active->success(), "an accepted stage may finish after its submission deadline");
    require(queued->done() && queued->result().buffer_safe
                && queued->errorInfo().code() == ErrorCode::DEADLINE_EXCEEDED
                && submitted_batches.load(std::memory_order_relaxed) == 1,
            "the queue must expire the second batch without submitting I/O");
}
}  // namespace

int main() {
    try {
        boundedBatchesKeepResultOrder();
        admissionRetainsCapacityUntilCompletion();
        expiredWorkDoesNotReachIo();
        queuedDeadlineDoesNotRevokeSubmittedIo();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
