#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/connector/p2p/DecodeTargetWriteLease.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerDecode.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {
namespace test {

// =============================================================================
// InflightMockRecvTask: simulates real TransferTask cancel semantics.
//
// Key difference from MockIKVCacheRecvTask:
//   - cancel() on a TRANSFERRING task only sets cancel_requested_, does NOT
//     set done_=true. The task remains in-flight until notifyDone() is called
//     externally (simulating the transport layer completing the transfer).
//   - This accurately models the race window: after cancel(), done()==false
//     until the transport layer finishes.
// =============================================================================
class InflightMockRecvTask: public transfer::IKVCacheRecvTask {
public:
    bool done() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_;
    }

    bool success() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_ && error_code_ == transfer::TransferErrorCode::OK;
    }

    void cancel() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_)
            return;
        if (!transferring_) {
            done_       = true;
            error_code_ = transfer::TransferErrorCode::CANCELLED;
        } else {
            cancel_requested_ = true;
        }
        cv_.notify_all();
    }

    void forceCancel() override {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_)
            return;
        done_       = true;
        error_code_ = transfer::TransferErrorCode::CANCELLED;
        cv_.notify_all();
    }

    transfer::TransferErrorCode errorCode() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return error_code_;
    }

    std::string errorMessage() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_ ? (error_code_ == transfer::TransferErrorCode::OK ? "" : "inflight mock error") : "";
    }

    // --- Test control API ---

    void startTransfer() {
        std::lock_guard<std::mutex> lock(mutex_);
        transferring_ = true;
    }

    void notifyDone(bool ok = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (done_)
            return;
        done_ = true;
        if (cancel_requested_) {
            error_code_ = transfer::TransferErrorCode::CANCELLED;
        } else {
            error_code_ = ok ? transfer::TransferErrorCode::OK : transfer::TransferErrorCode::UNKNOWN;
        }
        cv_.notify_all();
    }

    bool isTransferring() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return transferring_ && !done_;
    }

    bool isCancelRequested() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return cancel_requested_;
    }

    void waitUntilCancelRequested(int timeout_ms = 2000) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return cancel_requested_ || done_; });
    }

private:
    mutable std::mutex          mutex_;
    std::condition_variable     cv_;
    bool                        done_{false};
    bool                        transferring_{false};
    bool                        cancel_requested_{false};
    transfer::TransferErrorCode error_code_{transfer::TransferErrorCode::OK};
};

// =============================================================================
// InflightMockReceiver: creates InflightMockRecvTask instances.
// All created tasks default to TRANSFERRING state (simulating the common case
// where transport has already started when cancel arrives).
// =============================================================================
class InflightMockReceiver: public transfer::IKVCacheReceiver {
public:
    bool regMem(const BlockInfo&, uint64_t) override {
        return true;
    }

    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override {
        auto task = std::make_shared<InflightMockRecvTask>();
        if (start_in_transferring_) {
            task->startTransfer();
        }
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_[request.unique_key] = task;
        task_creation_order_.push_back(request.unique_key);
        cv_.notify_all();
        return task;
    }

    void stealTask(const std::string&) override {
        steal_count_.fetch_add(1);
    }

    transfer::IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = tasks_.find(unique_key);
        return it != tasks_.end() ? it->second : nullptr;
    }

    // --- Test control API ---

    std::shared_ptr<InflightMockRecvTask> getInflightTask(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        it = tasks_.find(key);
        return it != tasks_.end() ? it->second : nullptr;
    }

    std::vector<std::shared_ptr<InflightMockRecvTask>> getAllTasks() {
        std::lock_guard<std::mutex>                        lock(mutex_);
        std::vector<std::shared_ptr<InflightMockRecvTask>> result;
        for (auto& [k, v] : tasks_) {
            result.push_back(v);
        }
        return result;
    }

    std::vector<std::string> getTaskKeys() {
        std::lock_guard<std::mutex> lock(mutex_);
        return task_creation_order_;
    }

    void completeAllTasks(bool ok = true) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [k, task] : tasks_) {
            task->notifyDone(ok);
        }
    }

    void waitForTaskCount(int count, int timeout_ms = 2000) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(
            lock, std::chrono::milliseconds(timeout_ms), [&] { return static_cast<int>(tasks_.size()) >= count; });
    }

    void setStartInTransferring(bool v) {
        start_in_transferring_ = v;
    }

    int stealCount() const {
        return steal_count_.load();
    }

private:
    mutable std::mutex                                                     mutex_;
    std::condition_variable                                                cv_;
    std::unordered_map<std::string, std::shared_ptr<InflightMockRecvTask>> tasks_;
    std::vector<std::string>                                               task_creation_order_;
    std::atomic<int>                                                       steal_count_{0};
    bool                                                                   start_in_transferring_{true};
};

// =============================================================================
// MockLayerBlockConverter (minimal)
// =============================================================================
class LeaseTestMockLayerBlockConverter: public LayerBlockConverter {
public:
    std::vector<BlockInfo> convertIndexToBuffer(int, int, int, int) const override {
        return {};
    }
    std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const override {
        return {};
    }
};

// =============================================================================
// Test fixture for Decode Worker Lease Race Condition tests.
// =============================================================================
class DecodeLeaseRaceTest: public ::testing::Test {
protected:
    void SetUp() override {
        config_.tp_size                            = 1;
        config_.tp_rank                            = 0;
        config_.layer_all_num                      = 2;
        config_.p2p_read_return_before_deadline_ms = 50;
        config_.p2p_read_steal_before_deadline_ms  = 80;

        mock_converter_    = std::make_shared<LeaseTestMockLayerBlockConverter>();
        inflight_receiver_ = std::make_shared<InflightMockReceiver>();

        decode_ = std::make_unique<P2PConnectorWorkerDecode>(config_, mock_converter_, nullptr, inflight_receiver_);
    }

    void TearDown() override {
        decode_.reset();
    }

    std::shared_ptr<LayerCacheBuffer> makeBuffer(int layer_id, int num_blocks = 2) {
        auto buf = std::make_shared<LayerCacheBuffer>(layer_id);
        for (int i = 0; i < num_blocks; ++i) {
            buf->addBlockId(layer_id * 1000 + i, i);
        }
        return buf;
    }

    std::vector<std::shared_ptr<LayerCacheBuffer>> makeBuffers(int num_layers = 2, int blocks_per_layer = 2) {
        std::vector<std::shared_ptr<LayerCacheBuffer>> bufs;
        for (int i = 0; i < num_layers; ++i) {
            bufs.push_back(makeBuffer(i, blocks_per_layer));
        }
        return bufs;
    }

    // Helper: run read() in a thread and return the thread + result holder
    struct ReadResult {
        ErrorInfo         error;
        std::atomic<bool> done{false};
    };

    std::thread startReadThread(const std::string&                                    key,
                                int64_t                                               deadline_ms,
                                const std::vector<std::shared_ptr<LayerCacheBuffer>>& buffers,
                                std::shared_ptr<ReadResult>                           result,
                                int                                                   remote_tp_size = 1) {
        return std::thread([this, key, deadline_ms, buffers, result, remote_tp_size]() {
            result->error = decode_->read(1, key, deadline_ms, buffers, remote_tp_size);
            result->done  = true;
        });
    }

protected:
    P2PConnectorWorkerConfig                          config_;
    std::shared_ptr<LeaseTestMockLayerBlockConverter> mock_converter_;
    std::shared_ptr<InflightMockReceiver>             inflight_receiver_;
    std::unique_ptr<P2PConnectorWorkerDecode>         decode_;
};

// =============================================================================
// GROUP A: Cancelled-path lease lifecycle
//
// The bug: when read() exits via Cancelled, it immediately erases lease_map_,
// even when TRANSFERRING tasks haven't finished. queryLeaseStatus then returns
// "not found" → treats as stopped → blocks freed prematurely.
// =============================================================================

// A1: After cancel with in-flight transfers, queryLeaseStatus must still report
//     the lease as NOT stopped until all tasks complete.
TEST_F(DecodeLeaseRaceTest, A1_CancelWithInflightTransfers_LeaseNotStoppedUntilDone) {
    const std::string key         = "a1_cancel_inflight";
    auto              buffers     = makeBuffers(2);
    int64_t           deadline_ms = currentTimeMs() + 5000;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    // Wait for tasks to be created
    inflight_receiver_->waitForTaskCount(2);

    // Cancel while tasks are in-flight (TRANSFERRING state)
    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    // Wait for read() to return (it sees cancelled flag)
    // Note: with InflightMockRecvTask, cancel() on TRANSFERRING doesn't make done()=true,
    // BUT the existing code's MockIKVCacheRecvTask DOES make it done immediately.
    // The existing implementation only checks cancelled flag in the loop, not done() of all tasks.
    // So read() will return with Cancelled outcome once it detects cancelled flag.
    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());
    EXPECT_EQ(result->error.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED);

    // NOW THE CRITICAL CHECK: tasks are still in-flight (done()==false because TRANSFERRING)
    auto tasks          = inflight_receiver_->getAllTasks();
    int  inflight_count = 0;
    for (auto& t : tasks) {
        if (!t->done())
            ++inflight_count;
    }

    // BUG DETECTION: queryLeaseStatus should still find the lease and report NOT stopped.
    // With the bug, lease_map_ is already erased → queryLeaseStatus returns false (not found)
    // which the caller interprets as "stopped" → premature block free!
    bool sealed = false, stopped = false;
    int  started_ops = 0, finished_ops = 0;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);

    if (inflight_count > 0) {
        // If transfers are still in-flight, the lease MUST still be in the map
        EXPECT_TRUE(found) << "BUG: lease_map_ erased while " << inflight_count
                           << " transfers still in-flight. queryLeaseStatus returns 'not found' → "
                              "caller treats as stopped → premature block free!";
        if (found) {
            EXPECT_TRUE(sealed);
            EXPECT_FALSE(stopped) << "BUG: lease reports stopped but " << inflight_count
                                  << " transfers still in-flight";
            EXPECT_GT(started_ops, finished_ops);
        }
    }

    // Now complete all in-flight tasks
    inflight_receiver_->completeAllTasks(true);

    // After all tasks complete, queryLeaseStatus should eventually report stopped
    // (or be cleaned up from the map)
    bool eventually_stopped = false;
    for (int i = 0; i < 50; ++i) {
        found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
        if (!found || stopped) {
            eventually_stopped = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(eventually_stopped) << "Lease never became stopped after all transfers completed (leak)";

    if (read_thread.joinable())
        read_thread.join();
}

// A2: Cancel when ALL tasks happen to be in PENDING state (not yet TRANSFERRING).
//     cancel() makes them done immediately → lease should be stopped right away.
TEST_F(DecodeLeaseRaceTest, A2_CancelAllPending_LeaseStoppedImmediately) {
    // Tasks are NOT in transferring state
    inflight_receiver_->setStartInTransferring(false);

    const std::string key         = "a2_cancel_pending";
    auto              buffers     = makeBuffers(2);
    int64_t           deadline_ms = currentTimeMs() + 5000;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(2);

    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // All tasks were PENDING → cancel() made them done immediately
    // So it's acceptable for lease_map_ to be erased (all ops are finished)
    auto tasks = inflight_receiver_->getAllTasks();
    for (auto& t : tasks) {
        EXPECT_TRUE(t->done()) << "PENDING task should be done after cancel()";
    }

    // queryLeaseStatus: lease either cleaned up (not found → treated as stopped) or stopped=true
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    if (found) {
        EXPECT_TRUE(stopped) << "All tasks done, lease should be stopped";
    }
    // Not found is also acceptable (lease cleaned up because all done)

    if (read_thread.joinable())
        read_thread.join();
}

// A3: Cancel with mixed states - some tasks PENDING, some TRANSFERRING.
//     Lease must not be marked stopped until the TRANSFERRING tasks finish.
TEST_F(DecodeLeaseRaceTest, A3_CancelMixedPendingAndTransferring_LeaseWaitsForInflight) {
    // We need more tasks to have a mix. Use 4 layers.
    config_.layer_all_num = 4;
    decode_ = std::make_unique<P2PConnectorWorkerDecode>(config_, mock_converter_, nullptr, inflight_receiver_);

    // Start all in PENDING, then manually transition some to TRANSFERRING
    inflight_receiver_->setStartInTransferring(false);

    const std::string key         = "a3_mixed_states";
    auto              buffers     = makeBuffers(4);
    int64_t           deadline_ms = currentTimeMs() + 5000;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(4);

    // Transition tasks for layer 0 and 1 to TRANSFERRING state
    auto task_keys = inflight_receiver_->getTaskKeys();
    ASSERT_GE(task_keys.size(), 4u);
    auto task0 = inflight_receiver_->getInflightTask(task_keys[0]);
    auto task1 = inflight_receiver_->getInflightTask(task_keys[1]);
    task0->startTransfer();
    task1->startTransfer();
    // tasks 2 and 3 remain PENDING

    // Cancel
    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // PENDING tasks (2,3) should be done (cancel makes them done immediately)
    auto task2 = inflight_receiver_->getInflightTask(task_keys[2]);
    auto task3 = inflight_receiver_->getInflightTask(task_keys[3]);
    EXPECT_TRUE(task2->done());
    EXPECT_TRUE(task3->done());

    // TRANSFERRING tasks (0,1) should NOT be done yet
    EXPECT_FALSE(task0->done()) << "TRANSFERRING task should not be done after cancel()";
    EXPECT_FALSE(task1->done()) << "TRANSFERRING task should not be done after cancel()";

    // queryLeaseStatus: lease must NOT be stopped while task0 and task1 are in-flight
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);

    EXPECT_TRUE(found) << "BUG: lease_map_ erased while 2 TRANSFERRING tasks still in-flight";
    if (found) {
        EXPECT_TRUE(sealed);
        EXPECT_FALSE(stopped) << "BUG: lease reports stopped but TRANSFERRING tasks 0,1 are still in-flight";
    }

    // Complete task0 only
    task0->notifyDone(true);
    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    if (found) {
        EXPECT_FALSE(stopped) << "task1 still in-flight, should not be stopped";
    }

    // Complete task1
    task1->notifyDone(true);
    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    // Now all tasks are done → should be stopped (or already cleaned up)
    if (found) {
        EXPECT_TRUE(stopped) << "All tasks done, lease should be stopped now";
    }

    if (read_thread.joinable())
        read_thread.join();
}

// =============================================================================
// GROUP B: TRANSFER_NOT_DONE path (return deadline reached with pending transfers)
//
// This path correctly keeps lease_map_ entry alive. Verify queryLeaseStatus
// works as expected in this scenario.
// =============================================================================

// B1: read() times out (TRANSFER_NOT_DONE), tasks complete later via queryLeaseStatus polling.
TEST_F(DecodeLeaseRaceTest, B1_TransferNotDone_LeasePolledUntilAllComplete) {
    const std::string key     = "b1_transfer_not_done";
    auto              buffers = makeBuffers(2);
    // Very short deadline so read() hits return_deadline quickly
    int64_t deadline_ms = currentTimeMs() + 60;  // return_deadline = deadline - 50 = now+10ms

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(2);

    // Don't complete tasks - let read() time out
    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());
    EXPECT_EQ(result->error.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    // Lease must be in the map (TRANSFER_NOT_DONE path keeps it)
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_TRUE(found);
    EXPECT_TRUE(sealed);
    EXPECT_FALSE(stopped) << "Tasks still in-flight, should not be stopped";
    EXPECT_EQ(started_ops, 2);
    EXPECT_EQ(finished_ops, 0);

    // Complete one task
    auto task_keys = inflight_receiver_->getTaskKeys();
    auto task0     = inflight_receiver_->getInflightTask(task_keys[0]);
    task0->notifyDone(true);

    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_TRUE(found);
    EXPECT_FALSE(stopped);
    EXPECT_EQ(finished_ops, 1);

    // Complete second task
    auto task1 = inflight_receiver_->getInflightTask(task_keys[1]);
    task1->notifyDone(true);

    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    // Should be stopped now (and lazily removed from map)
    // After this call, the entry is erased, so found may be true with stopped=true
    if (found) {
        EXPECT_TRUE(stopped);
        EXPECT_EQ(started_ops, 2);
        EXPECT_EQ(finished_ops, 2);
    }

    // Subsequent query: not found (already cleaned up)
    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_FALSE(found);
    EXPECT_TRUE(stopped);  // returns stopped=true when not found

    if (read_thread.joinable())
        read_thread.join();
}

// B2: TRANSFER_NOT_DONE with tasks that were cancel-requested during wait
//     (steal happened, cancel_requested_ set). Tasks should still complete normally.
TEST_F(DecodeLeaseRaceTest, B2_TransferNotDone_StolenTasksStillComplete) {
    const std::string key         = "b2_stolen_tasks";
    auto              buffers     = makeBuffers(2);
    int64_t           deadline_ms = currentTimeMs() + 60;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(2);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());
    EXPECT_EQ(result->error.code(), ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE);

    // Verify steal was called (steal_before > return_before, so steal happens before return)
    EXPECT_GE(inflight_receiver_->stealCount(), 1);

    // Tasks complete with cancel_requested (simulating transport layer finishing)
    auto tasks = inflight_receiver_->getAllTasks();
    for (auto& t : tasks) {
        t->notifyDone(true);  // Transport layer reports success (cancel_requested is just a hint)
    }

    // Lease should be stopped now
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    if (found) {
        EXPECT_TRUE(stopped);
    }

    if (read_thread.joinable())
        read_thread.join();
}

// =============================================================================
// GROUP D: queryLeaseStatus driven finish counting
//
// Verify the lazy counting mechanism works correctly.
// =============================================================================

// D1: queryLeaseStatus correctly counts newly-done tasks incrementally.
TEST_F(DecodeLeaseRaceTest, D1_QueryLeaseStatus_IncrementalFinishCounting) {
    const std::string key     = "d1_incremental";
    auto              buffers = makeBuffers(3);
    config_.layer_all_num     = 3;
    decode_ = std::make_unique<P2PConnectorWorkerDecode>(config_, mock_converter_, nullptr, inflight_receiver_);

    int64_t deadline_ms = currentTimeMs() + 60;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(3);

    // Let read() time out (TRANSFER_NOT_DONE)
    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    auto task_keys = inflight_receiver_->getTaskKeys();
    ASSERT_EQ(task_keys.size(), 3u);

    // Query: 0 finished
    bool sealed, stopped;
    int  started_ops, finished_ops;
    decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(started_ops, 3);
    EXPECT_EQ(finished_ops, 0);
    EXPECT_FALSE(stopped);

    // Complete task 0
    inflight_receiver_->getInflightTask(task_keys[0])->notifyDone(true);
    decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(finished_ops, 1);
    EXPECT_FALSE(stopped);

    // Complete task 1
    inflight_receiver_->getInflightTask(task_keys[1])->notifyDone(true);
    decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(finished_ops, 2);
    EXPECT_FALSE(stopped);

    // Complete task 2 → all done → stopped
    inflight_receiver_->getInflightTask(task_keys[2])->notifyDone(true);
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(finished_ops, 3);
    EXPECT_TRUE(stopped);

    // After stopped, entry is lazily removed
    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_FALSE(found);

    if (read_thread.joinable())
        read_thread.join();
}

// D2: Multiple calls to queryLeaseStatus with no new completions between calls.
//     finish_counted must not double-count.
TEST_F(DecodeLeaseRaceTest, D2_QueryLeaseStatus_NoDoubleCount) {
    const std::string key         = "d2_no_double_count";
    auto              buffers     = makeBuffers(2);
    int64_t           deadline_ms = currentTimeMs() + 60;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(2);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // Complete one task
    auto task_keys = inflight_receiver_->getTaskKeys();
    inflight_receiver_->getInflightTask(task_keys[0])->notifyDone(true);

    bool sealed, stopped;
    int  started_ops, finished_ops;

    // First query: should count 1 done
    decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(finished_ops, 1);

    // Second query without any new completions: must still report 1, not 2
    decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(finished_ops, 1);

    // Third query: still 1
    decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(finished_ops, 1);
    EXPECT_FALSE(stopped);

    // Now complete second task
    inflight_receiver_->getInflightTask(task_keys[1])->notifyDone(true);
    decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    EXPECT_EQ(finished_ops, 2);
    EXPECT_TRUE(stopped);

    if (read_thread.joinable())
        read_thread.join();
}

// =============================================================================
// GROUP E: Multi-partition (multi-TP) cancel races
//
// With remote_tp_size > local_tp_size, each layer creates multiple recv tasks
// (one per partition). Cancel must wait for ALL partitions to complete.
// =============================================================================

// E1: 2P1D (remote_tp=2, local_tp=1 → 2 partitions per layer).
//     Cancel with one partition done and one still in-flight.
TEST_F(DecodeLeaseRaceTest, E1_MultiTP_CancelOnePartitionDoneOneInflight) {
    const std::string key            = "e1_multi_partition";
    auto              buffers        = makeBuffers(1);  // 1 layer, but 2 partitions
    int64_t           deadline_ms    = currentTimeMs() + 5000;
    int               remote_tp_size = 2;  // → recv_partition_count = 2/1 = 2

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result, remote_tp_size);

    // Should create 2 tasks (1 layer × 2 partitions)
    inflight_receiver_->waitForTaskCount(2);
    auto task_keys = inflight_receiver_->getTaskKeys();
    ASSERT_EQ(task_keys.size(), 2u);

    auto partition0 = inflight_receiver_->getInflightTask(task_keys[0]);
    auto partition1 = inflight_receiver_->getInflightTask(task_keys[1]);

    // Complete partition 0
    partition0->notifyDone(true);

    // Cancel while partition 1 is still in-flight
    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // Partition 1 is still in-flight
    EXPECT_FALSE(partition1->done());

    // Lease should NOT be stopped
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);

    EXPECT_TRUE(found) << "BUG: lease erased while partition 1 still in-flight";
    if (found) {
        EXPECT_TRUE(sealed);
        EXPECT_FALSE(stopped);
        EXPECT_EQ(started_ops, 2);
    }

    // Complete partition 1
    partition1->notifyDone(true);
    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    if (found) {
        EXPECT_TRUE(stopped);
        EXPECT_EQ(finished_ops, 2);
    }

    if (read_thread.joinable())
        read_thread.join();
}

// E2: 4P1D (4 partitions per layer), cancel with 3 done and 1 in-flight.
//     The last in-flight partition prevents lease from being stopped.
TEST_F(DecodeLeaseRaceTest, E2_MultiTP_4Partitions_LastOneInflight) {
    const std::string key            = "e2_4partitions";
    auto              buffers        = makeBuffers(1);
    int64_t           deadline_ms    = currentTimeMs() + 5000;
    int               remote_tp_size = 4;  // → 4 partitions

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result, remote_tp_size);

    inflight_receiver_->waitForTaskCount(4);
    auto task_keys = inflight_receiver_->getTaskKeys();
    ASSERT_EQ(task_keys.size(), 4u);

    // Complete first 3 partitions
    for (int i = 0; i < 3; ++i) {
        inflight_receiver_->getInflightTask(task_keys[i])->notifyDone(true);
    }

    // Cancel
    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // Last partition still in-flight
    auto last_task = inflight_receiver_->getInflightTask(task_keys[3]);
    EXPECT_FALSE(last_task->done());

    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);

    EXPECT_TRUE(found) << "BUG: lease erased while partition 3 still in-flight";
    if (found) {
        EXPECT_FALSE(stopped);
        EXPECT_EQ(started_ops, 4);
    }

    // Complete last partition
    last_task->notifyDone(true);
    found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    if (found) {
        EXPECT_TRUE(stopped);
        EXPECT_EQ(finished_ops, 4);
    }

    if (read_thread.joinable())
        read_thread.join();
}

// E3: Multi-layer + multi-partition: 2 layers × 2 partitions = 4 tasks.
//     Cancel with tasks completing staggered.
TEST_F(DecodeLeaseRaceTest, E3_MultiLayerMultiPartition_StaggeredCompletion) {
    config_.layer_all_num = 2;
    decode_ = std::make_unique<P2PConnectorWorkerDecode>(config_, mock_converter_, nullptr, inflight_receiver_);

    const std::string key            = "e3_multi_layer_multi_part";
    auto              buffers        = makeBuffers(2);
    int64_t           deadline_ms    = currentTimeMs() + 5000;
    int               remote_tp_size = 2;  // 2 layers × 2 partitions = 4 tasks

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result, remote_tp_size);

    inflight_receiver_->waitForTaskCount(4);
    auto task_keys = inflight_receiver_->getTaskKeys();
    ASSERT_EQ(task_keys.size(), 4u);

    // Cancel immediately (all tasks in-flight)
    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // Complete tasks one by one, verify lease tracks correctly
    bool sealed, stopped;
    int  started_ops, finished_ops;

    for (int i = 0; i < 4; ++i) {
        inflight_receiver_->getInflightTask(task_keys[i])->notifyDone(true);
        bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);

        if (i < 3) {
            // Not all done yet
            EXPECT_TRUE(found) << "Lease should still exist at completion " << (i + 1) << "/4";
            if (found) {
                EXPECT_EQ(finished_ops, i + 1);
                EXPECT_FALSE(stopped);
            }
        } else {
            // All done
            if (found) {
                EXPECT_EQ(finished_ops, 4);
                EXPECT_TRUE(stopped);
            }
        }
    }

    if (read_thread.joinable())
        read_thread.join();
}

// =============================================================================
// GROUP F: Concurrent cancel + completion race
//
// Tests where cancel and transport-layer completion happen simultaneously.
// =============================================================================

// F1: Task completes (notifyDone) at the exact moment cancel is issued.
//     No matter who wins, lease must reach stopped eventually.
TEST_F(DecodeLeaseRaceTest, F1_ConcurrentCancelAndCompletion) {
    const std::string key         = "f1_concurrent";
    auto              buffers     = makeBuffers(1);
    int64_t           deadline_ms = currentTimeMs() + 5000;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(1);
    auto task_keys = inflight_receiver_->getTaskKeys();
    auto task      = inflight_receiver_->getInflightTask(task_keys[0]);

    // Race: complete and cancel simultaneously
    std::thread completer([&task]() { task->notifyDone(true); });

    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    completer.join();

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // Task should be done regardless of race outcome
    EXPECT_TRUE(task->done());

    // Lease must eventually reach stopped (no leak)
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool eventually_stopped = false;
    for (int i = 0; i < 50; ++i) {
        bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
        if (!found || stopped) {
            eventually_stopped = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    EXPECT_TRUE(eventually_stopped) << "Lease never reached stopped state (potential leak)";

    if (read_thread.joinable())
        read_thread.join();
}

// F2: All tasks complete BEFORE cancel takes effect (cancel arrives too late).
//     read() returns AllDone normally.
TEST_F(DecodeLeaseRaceTest, F2_AllTasksCompletedBeforeCancel) {
    const std::string key         = "f2_complete_before_cancel";
    auto              buffers     = makeBuffers(2);
    int64_t           deadline_ms = currentTimeMs() + 5000;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(2);

    // Complete all tasks immediately
    inflight_receiver_->completeAllTasks(true);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());
    // read() exits via AllDone path (cancel never got a chance)
    EXPECT_TRUE(result->error.ok());

    // Lease should be cleaned up (AllDone path seals and erases)
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    // Not found means already cleaned up → OK
    if (found) {
        EXPECT_TRUE(stopped);
    }

    if (read_thread.joinable())
        read_thread.join();
}

// F3: Delayed notifyDone simulating network latency after cancel.
//     Verifies that even with significant delay, lease properly transitions to stopped.
TEST_F(DecodeLeaseRaceTest, F3_DelayedNotifyDoneAfterCancel) {
    const std::string key         = "f3_delayed_notify";
    auto              buffers     = makeBuffers(2);
    int64_t           deadline_ms = currentTimeMs() + 5000;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(2);

    // Cancel immediately
    bool cancelled = false;
    for (int i = 0; i < 100 && !cancelled; ++i) {
        cancelled = decode_->cancelRead(key);
        if (!cancelled)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    ASSERT_TRUE(cancelled);

    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    // Simulate network latency: tasks complete 100ms after cancel
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    auto tasks          = inflight_receiver_->getAllTasks();
    int  still_inflight = 0;
    for (auto& t : tasks) {
        if (!t->done())
            ++still_inflight;
    }

    if (still_inflight > 0) {
        // Verify lease is still alive
        bool sealed, stopped;
        int  started_ops, finished_ops;
        bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
        EXPECT_TRUE(found) << "BUG: lease gone while " << still_inflight << " tasks still in-flight after 100ms delay";
        if (found) {
            EXPECT_FALSE(stopped);
        }
    }

    // Now complete all tasks
    for (auto& t : tasks) {
        if (!t->done())
            t->notifyDone(true);
    }

    // Verify stopped
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    if (found) {
        EXPECT_TRUE(stopped);
    }

    if (read_thread.joinable())
        read_thread.join();
}

// F4: Stress test - rapid repeated queryLeaseStatus during async completion.
TEST_F(DecodeLeaseRaceTest, F4_RapidQueryDuringAsyncCompletion) {
    const std::string key         = "f4_rapid_query";
    auto              buffers     = makeBuffers(2);
    int64_t           deadline_ms = currentTimeMs() + 60;

    auto result      = std::make_shared<ReadResult>();
    auto read_thread = startReadThread(key, deadline_ms, buffers, result);

    inflight_receiver_->waitForTaskCount(2);

    // Wait for TRANSFER_NOT_DONE
    int wait = 0;
    while (!result->done && wait < 200) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        ++wait;
    }
    ASSERT_TRUE(result->done.load());

    auto task_keys = inflight_receiver_->getTaskKeys();

    // Start a thread that queries rapidly
    std::atomic<bool> query_stop{false};
    std::atomic<int>  max_finished_seen{0};
    std::thread       query_thread([&]() {
        while (!query_stop) {
            bool sealed, stopped;
            int  started, finished;
            decode_->queryLeaseStatus(key, sealed, started, finished, stopped);
            int cur_max = max_finished_seen.load();
            while (finished > cur_max && !max_finished_seen.compare_exchange_weak(cur_max, finished))
                ;
            if (stopped)
                break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    // Complete tasks with small delays
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    inflight_receiver_->getInflightTask(task_keys[0])->notifyDone(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    inflight_receiver_->getInflightTask(task_keys[1])->notifyDone(true);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    query_stop = true;
    query_thread.join();

    // Verify finished count monotonically increased and reached 2
    EXPECT_EQ(max_finished_seen.load(), 2);

    // Final check: lease is stopped
    bool sealed, stopped;
    int  started_ops, finished_ops;
    bool found = decode_->queryLeaseStatus(key, sealed, started_ops, finished_ops, stopped);
    if (found) {
        EXPECT_TRUE(stopped);
    }

    if (read_thread.joinable())
        read_thread.join();
}

// =============================================================================
// GROUP G: DecodeTargetWriteLease unit tests (standalone, no Worker involved)
// =============================================================================

TEST(DecodeTargetWriteLeaseTest, InitialState_NotSealedNotStopped) {
    DecodeTargetWriteLease lease;
    EXPECT_FALSE(lease.isSealed());
    EXPECT_FALSE(lease.isStopped());
    EXPECT_EQ(lease.startedOps(), 0);
    EXPECT_EQ(lease.finishedOps(), 0);
}

TEST(DecodeTargetWriteLeaseTest, SealWithZeroOps_ImmediatelyStopped) {
    DecodeTargetWriteLease lease;
    lease.seal();
    EXPECT_TRUE(lease.isSealed());
    EXPECT_TRUE(lease.isStopped());
}

TEST(DecodeTargetWriteLeaseTest, SealWithPendingOps_NotStoppedUntilAllFinish) {
    DecodeTargetWriteLease lease;
    lease.onTransferStarted();
    lease.onTransferStarted();
    lease.onTransferStarted();
    lease.seal();

    EXPECT_TRUE(lease.isSealed());
    EXPECT_FALSE(lease.isStopped());
    EXPECT_EQ(lease.startedOps(), 3);
    EXPECT_EQ(lease.finishedOps(), 0);

    lease.onTransferFinished();
    EXPECT_FALSE(lease.isStopped());

    lease.onTransferFinished();
    EXPECT_FALSE(lease.isStopped());

    lease.onTransferFinished();
    EXPECT_TRUE(lease.isStopped());
    EXPECT_EQ(lease.finishedOps(), 3);
}

TEST(DecodeTargetWriteLeaseTest, FinishBeforeSeal_StoppedOnceSealCalled) {
    DecodeTargetWriteLease lease;
    lease.onTransferStarted();
    lease.onTransferFinished();

    EXPECT_FALSE(lease.isStopped());  // not sealed yet

    lease.seal();
    EXPECT_TRUE(lease.isStopped());  // now sealed and started==finished
}

TEST(DecodeTargetWriteLeaseTest, ConcurrentStartAndFinish_EventuallyConverges) {
    DecodeTargetWriteLease lease;
    constexpr int          N = 100;

    std::vector<std::thread> starters, finishers;
    for (int i = 0; i < N; ++i) {
        starters.emplace_back([&lease]() { lease.onTransferStarted(); });
    }
    for (auto& t : starters)
        t.join();

    EXPECT_EQ(lease.startedOps(), N);

    lease.seal();
    EXPECT_FALSE(lease.isStopped());

    for (int i = 0; i < N; ++i) {
        finishers.emplace_back([&lease]() { lease.onTransferFinished(); });
    }
    for (auto& t : finishers)
        t.join();

    EXPECT_TRUE(lease.isStopped());
    EXPECT_EQ(lease.finishedOps(), N);
}

}  // namespace test
}  // namespace rtp_llm
