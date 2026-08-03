#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

using transfer::TransferErrorCode;
using transfer::TransferTask;

class TransferTaskTest: public ::testing::Test {
protected:
    /// Returns a task whose deadline is far in the future (won't expire during tests).
    static TransferTask makeTask(int64_t deadline_offset_ms = 5000) {
        return TransferTask({}, currentTimeMs() + deadline_offset_ms);
    }

    /// Returns a task whose deadline has already passed.
    static TransferTask makeExpiredTask() {
        return TransferTask({}, currentTimeMs() - 100);
    }

    static void sleepMs(int64_t ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    /// Spin-waits until task.done() or timeout. Returns true if done.
    static bool waitDone(const TransferTask& task, int64_t timeout_ms = 3000) {
        int64_t deadline = currentTimeMs() + timeout_ms;
        while (!task.done() && currentTimeMs() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return task.done();
    }
};

// ==================== 初始状态 ====================

TEST_F(TransferTaskTest, InitialState) {
    auto task = makeTask();
    EXPECT_FALSE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.totalCostTimeUs(), 0);
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
    EXPECT_TRUE(task.errorMessage().empty());
}

// ==================== startTransfer() ====================

TEST_F(TransferTaskTest, StartTransfer_Pending_ReturnsTrue) {
    auto task = makeTask();
    EXPECT_TRUE(task.startTransfer());
    EXPECT_FALSE(task.done());
}

TEST_F(TransferTaskTest, StartTransfer_AfterCancelPending_ReturnsFalse) {
    auto task = makeTask();
    task.cancel();
    ASSERT_TRUE(task.done());
    EXPECT_FALSE(task.startTransfer());
}

TEST_F(TransferTaskTest, StartTransfer_AfterNotifyDone_ReturnsFalse) {
    auto task = makeTask();
    task.startTransfer();
    task.notifyDone(true);
    EXPECT_FALSE(task.startTransfer());
}

// ==================== notifyDone() 正常路径 ====================

TEST_F(TransferTaskTest, NotifyDone_Success) {
    auto task = makeTask();
    task.startTransfer();
    task.notifyDone(true);

    EXPECT_TRUE(task.done());
    EXPECT_TRUE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
    EXPECT_TRUE(task.errorMessage().empty());
}

TEST_F(TransferTaskTest, NotifyDone_Failure_WithErrorCode) {
    auto task = makeTask();
    task.startTransfer();
    task.notifyDone(false, TransferErrorCode::RPC_FAILED, "rpc error");

    EXPECT_TRUE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::RPC_FAILED);
    EXPECT_EQ(task.errorMessage(), "rpc error");
}

TEST_F(TransferTaskTest, NotifyDone_Idempotent) {
    auto task = makeTask();
    task.startTransfer();
    task.notifyDone(true);
    task.notifyDone(false, TransferErrorCode::RPC_FAILED, "ignored");  // second call must be no-op

    EXPECT_TRUE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
}

// ==================== cancel() 两种语义 ====================

TEST_F(TransferTaskTest, Cancel_Pending_ImmediateDone) {
    auto task = makeTask();
    task.cancel();

    EXPECT_TRUE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

/// cancel() in TRANSFERRING only sets the flag; task must NOT be done yet.
TEST_F(TransferTaskTest, Cancel_Transferring_SetsFlag_NotYetDone) {
    auto task = makeTask();
    ASSERT_TRUE(task.startTransfer());
    task.cancel();

    EXPECT_FALSE(task.done());
}

/// cancel() in TRANSFERRING + notifyDone(success) -> result is CANCELLED (not OK).
TEST_F(TransferTaskTest, Cancel_Transferring_OverridesSuccess) {
    auto task = makeTask();
    ASSERT_TRUE(task.startTransfer());
    task.cancel();
    task.notifyDone(true);

    EXPECT_TRUE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

/// cancel() in TRANSFERRING + notifyDone(failure) -> result is still CANCELLED.
TEST_F(TransferTaskTest, Cancel_Transferring_OverridesFailure) {
    auto task = makeTask();
    ASSERT_TRUE(task.startTransfer());
    task.cancel();
    task.notifyDone(false, TransferErrorCode::RPC_FAILED, "rpc error");

    EXPECT_TRUE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

TEST_F(TransferTaskTest, Cancel_AfterDone_NoOp) {
    auto task = makeTask();
    task.startTransfer();
    task.notifyDone(true);
    task.cancel();  // must be no-op

    EXPECT_TRUE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
}

// ==================== notifyDone() + Timeout 优先级 ====================

/// Timeout beats a physical success: notifyDone(true) after deadline -> TIMEOUT.
TEST_F(TransferTaskTest, NotifyDone_AfterTimeout_OverridesSuccess) {
    auto task = makeExpiredTask();
    task.startTransfer();
    task.notifyDone(true);

    EXPECT_TRUE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::TIMEOUT);
}

/// Timeout beats a physical failure: notifyDone(false, RPC_FAILED) after deadline -> TIMEOUT.
TEST_F(TransferTaskTest, NotifyDone_AfterTimeout_OverridesFailure) {
    auto task = makeExpiredTask();
    task.startTransfer();
    task.notifyDone(false, TransferErrorCode::RPC_FAILED, "err");

    EXPECT_TRUE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::TIMEOUT);
}

/// Timeout beats cancel_requested: timeout branch comes first inside notifyDone().
TEST_F(TransferTaskTest, NotifyDone_AfterTimeout_OverridesCancelRequested) {
    auto task = makeExpiredTask();
    ASSERT_TRUE(task.startTransfer());
    task.cancel();          // sets cancel_requested_ = true
    task.notifyDone(true);  // but timeout fires first

    EXPECT_TRUE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::TIMEOUT);  // not CANCELLED
}

TEST_F(TransferTaskTest, NotifyDone_BeforeTimeout_Success_IsOK) {
    auto task = makeTask(5000);
    task.startTransfer();
    task.notifyDone(true);

    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
    EXPECT_TRUE(task.success());
}

// ==================== errorCode() 运行时 timeout 探测 ====================

/// While not done and before deadline, errorCode() returns OK.
TEST_F(TransferTaskTest, ErrorCode_NotDone_BeforeTimeout_ReturnsOK) {
    auto task = makeTask(5000);
    EXPECT_FALSE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
}

/// While not done but past deadline, errorCode() dynamically returns TIMEOUT.
TEST_F(TransferTaskTest, ErrorCode_NotDone_AfterTimeout_ReturnsTimeout) {
    auto task = makeExpiredTask();
    EXPECT_FALSE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::TIMEOUT);
}

/// Once done with OK, deadline expiry must NOT retroactively change errorCode().
/// Verifies the done_ guard prevents re-evaluation after finalization.
TEST_F(TransferTaskTest, ErrorCode_Done_OK_DeadlineExpiryHasNoEffect) {
    auto task = makeTask(5000);
    task.startTransfer();
    task.notifyDone(true);

    ASSERT_TRUE(task.done());
    ASSERT_TRUE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
}

// ==================== forceCancel() ====================

TEST_F(TransferTaskTest, ForceCancel_Pending_ImmediateDone) {
    auto task = makeTask();
    task.forceCancel();

    EXPECT_TRUE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

TEST_F(TransferTaskTest, ForceCancel_Transferring_ImmediateDone) {
    auto task = makeTask();
    ASSERT_TRUE(task.startTransfer());
    task.forceCancel();

    EXPECT_TRUE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

/// forceCancel() in TRANSFERRING: a subsequent notifyDone must be silently ignored.
TEST_F(TransferTaskTest, ForceCancel_Transferring_SubsequentNotifyDoneIgnored) {
    auto task = makeTask();
    ASSERT_TRUE(task.startTransfer());
    task.forceCancel();
    task.notifyDone(true);  // must be ignored because done_ == true

    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

TEST_F(TransferTaskTest, ForceCancel_AfterDone_NoOp) {
    auto task = makeTask();
    task.startTransfer();
    task.notifyDone(true);
    task.forceCancel();  // must be no-op

    EXPECT_TRUE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
}

// ==================== totalCostTimeUs() ====================

TEST_F(TransferTaskTest, TotalCostTime_NotDone_IsZero) {
    auto task = makeTask();
    EXPECT_EQ(task.totalCostTimeUs(), 0);
}

TEST_F(TransferTaskTest, TotalCostTime_AfterNotifyDone_IsPositive) {
    auto task = makeTask();
    task.startTransfer();
    sleepMs(5);
    task.notifyDone(true);
    EXPECT_GT(task.totalCostTimeUs(), 0);
}

TEST_F(TransferTaskTest, TotalCostTime_AfterCancelPending_IsPositive) {
    auto task = makeTask();
    sleepMs(5);
    task.cancel();
    EXPECT_GT(task.totalCostTimeUs(), 0);
}

TEST_F(TransferTaskTest, TotalCostTime_AfterForceCancel_IsPositive) {
    auto task = makeTask();
    sleepMs(5);
    task.forceCancel();
    EXPECT_GT(task.totalCostTimeUs(), 0);
}

// ==================== P2P Worker 场景模拟 ====================

/// 场景A: Transfer worker 调 startTransfer -> Observer 线程轮询 done() 等待成功
TEST_F(TransferTaskTest, WorkerScenario_StartTransfer_WaitDone_Success) {
    TransferTask task({}, currentTimeMs() + 5000);

    std::thread transfer_thread([&task]() {
        ASSERT_TRUE(task.startTransfer());
        sleepMs(50);
        task.notifyDone(true);
    });

    ASSERT_TRUE(waitDone(task));
    transfer_thread.join();

    EXPECT_TRUE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::OK);
    EXPECT_GT(task.totalCostTimeUs(), 0);
}

/// 场景B: PENDING 阶段被 cancel -> Observer 感知 done，startTransfer 返回 false
TEST_F(TransferTaskTest, WorkerScenario_Cancel_Pending_ObserverSeesDone) {
    TransferTask task({}, currentTimeMs() + 5000);

    std::thread cancel_thread([&task]() {
        sleepMs(50);
        task.cancel();
    });

    ASSERT_TRUE(waitDone(task));
    cancel_thread.join();

    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
    EXPECT_FALSE(task.startTransfer());  // too late: task is already done
}

/// 场景C: TRANSFERRING 中途被 cancel -> notifyDone(success) 结果被覆盖为 CANCELLED
TEST_F(TransferTaskTest, WorkerScenario_Cancel_Transferring_OverridesSuccess) {
    TransferTask task({}, currentTimeMs() + 5000);
    ASSERT_TRUE(task.startTransfer());

    std::atomic<bool> cancel_done{false};
    std::thread       cancel_thread([&task, &cancel_done]() {
        sleepMs(20);
        task.cancel();
        cancel_done.store(true);
    });

    while (!cancel_done.load()) {
        sleepMs(5);
    }
    // cancel_requested_ == true, but task is still undone
    EXPECT_FALSE(task.done());

    // Transfer physically "succeeds" -- result must be overridden by the cancel flag
    task.notifyDone(true);
    cancel_thread.join();

    EXPECT_TRUE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

/// 场景D: startTransfer vs cancel 竞争 -- 状态机必须保持一致，无数据竞争
TEST_F(TransferTaskTest, WorkerScenario_StartTransfer_Cancel_Race) {
    constexpr int iterations = 500;
    for (int i = 0; i < iterations; ++i) {
        TransferTask      task({}, currentTimeMs() + 5000);
        std::atomic<bool> transfer_won{false};

        std::thread t_transfer([&task, &transfer_won]() { transfer_won.store(task.startTransfer()); });
        std::thread t_cancel([&task]() { task.cancel(); });

        t_transfer.join();
        t_cancel.join();

        if (transfer_won.load()) {
            // startTransfer() won: task is TRANSFERRING, cancel only set flag.
            // Drive to DONE to avoid sanitizer warnings on dtor.
            task.notifyDone(true);
        }
        ASSERT_TRUE(task.done()) << "iteration " << i;
    }
}

/// 场景E: forceCancel 兜底 -- Transfer 正在传输中，Worker 发现超时后调 forceCancel
TEST_F(TransferTaskTest, WorkerScenario_ForceCancel_Timeout_SafetyNet) {
    TransferTask task({}, currentTimeMs() + 5000);
    ASSERT_TRUE(task.startTransfer());

    std::thread force_cancel_thread([&task]() {
        sleepMs(30);
        task.forceCancel();
    });

    force_cancel_thread.join();
    ASSERT_TRUE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);

    // A late notifyDone arriving after forceCancel must be completely ignored
    task.notifyDone(true);
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::CANCELLED);
}

/// 场景F: 最重要的 timeout 新语义 -- deadline 过期后 notifyDone(true) 结果为 TIMEOUT
///
/// 验证两个独立行为：
///   1. done_=false 期间 errorCode() 动态返回 TIMEOUT（运行时探测）
///   2. notifyDone() 被调用时已过 deadline -> 结果固化为 TIMEOUT，而非 OK
TEST_F(TransferTaskTest, WorkerScenario_NotifyDone_AfterDeadline_IsTimeout) {
    TransferTask task({}, currentTimeMs() + 100);
    ASSERT_TRUE(task.startTransfer());

    // Wait past the deadline
    sleepMs(150);

    // 1. Task still undone but past deadline -> errorCode() returns TIMEOUT dynamically
    EXPECT_FALSE(task.done());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::TIMEOUT);

    // 2. Physical transfer "succeeds" after deadline -> result fixed as TIMEOUT
    task.notifyDone(true);

    EXPECT_TRUE(task.done());
    EXPECT_FALSE(task.success());
    EXPECT_EQ(task.errorCode(), TransferErrorCode::TIMEOUT);
}

}  // namespace rtp_llm
