package org.flexlb.balance.autotpm;

import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.CancelReasonPB;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link RunningPreemptCommitter} — cancel + wait for release.
 */
class RunningPreemptCommitterTest {

    private static final String EP_KEY = "10.0.0.1:8080";

    // ==================== Cancel reason ====================

    @Test
    void cancelCalledWithPriorityPreempted_reason() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        PreemptRateLimiter rateLimiter = new PreemptRateLimiter(10, 50);

        List<CancelCall> cancelCalls = new ArrayList<>();
        // Cancel action that releases the reservation synchronously (mimics
        // FlexlbBatchScheduler.cancelRequest which calls rollbackOnce → release)
        BiConsumer<Long, CancelReasonPB> cancelAction = (reqId, reason) -> {
            cancelCalls.add(new CancelCall(reqId, reason));
            tracker.release(EP_KEY, reqId);
        };

        RunningPreemptCommitter committer = new RunningPreemptCommitter(
                cancelAction, tracker, rateLimiter);

        DecodeReservation victim = runningRes(10L, 30, 1000);
        tracker.reserve(EP_KEY, victim);

        boolean result = committer.execute(victim, EP_KEY, 2000L);

        assertTrue(result, "Preemption should succeed when release happens synchronously");
        assertEquals(1, cancelCalls.size(), "Cancel should be called exactly once");
        assertEquals(10L, cancelCalls.get(0).requestId);
        assertEquals(CancelReasonPB.CANCEL_REASON_PRIORITY_PREEMPTED,
                cancelCalls.get(0).reason,
                "Cancel reason must be CANCEL_REASON_PRIORITY_PREEMPTED");
    }

    // ==================== Rate limiting ====================

    @Test
    void rateLimited_returnsFalse_noCancelCalled() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // per-node limit = 0 → all preemptions rate-limited
        PreemptRateLimiter rateLimiter = new PreemptRateLimiter(0, 50);

        List<CancelCall> cancelCalls = new ArrayList<>();
        BiConsumer<Long, CancelReasonPB> cancelAction = (reqId, reason) -> {
            cancelCalls.add(new CancelCall(reqId, reason));
        };

        RunningPreemptCommitter committer = new RunningPreemptCommitter(
                cancelAction, tracker, rateLimiter);

        DecodeReservation victim = runningRes(10L, 30, 1000);
        tracker.reserve(EP_KEY, victim);

        boolean result = committer.execute(victim, EP_KEY, 2000L);

        assertFalse(result, "Should return false when rate-limited");
        assertTrue(cancelCalls.isEmpty(),
                "Cancel should NOT be called when rate-limited");
    }

    @Test
    void globalRateLimited_returnsFalse() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        // global limit = 0 → all preemptions rate-limited
        PreemptRateLimiter rateLimiter = new PreemptRateLimiter(10, 0);

        List<CancelCall> cancelCalls = new ArrayList<>();
        BiConsumer<Long, CancelReasonPB> cancelAction = (reqId, reason) -> {
            cancelCalls.add(new CancelCall(reqId, reason));
        };

        RunningPreemptCommitter committer = new RunningPreemptCommitter(
                cancelAction, tracker, rateLimiter);

        DecodeReservation victim = runningRes(10L, 30, 1000);
        tracker.reserve(EP_KEY, victim);

        boolean result = committer.execute(victim, EP_KEY, 2000L);

        assertFalse(result, "Should return false when global rate-limited");
        assertTrue(cancelCalls.isEmpty(),
                "Cancel should NOT be called when global rate-limited");
    }

    // ==================== Release within timeout ====================

    @Test
    void releaseWithinTimeout_returnsTrue() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        PreemptRateLimiter rateLimiter = new PreemptRateLimiter(10, 50);

        // Cancel action that releases asynchronously after a short delay,
        // simulating an engine that reports completion via WorkerStatus
        // on a separate thread.
        BiConsumer<Long, CancelReasonPB> cancelAction = (reqId, reason) -> {
            Thread t = new Thread(() -> {
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    return;
                }
                tracker.release(EP_KEY, reqId);
            });
            t.setDaemon(true);
            t.start();
        };

        RunningPreemptCommitter committer = new RunningPreemptCommitter(
                cancelAction, tracker, rateLimiter);

        DecodeReservation victim = runningRes(10L, 30, 1000);
        tracker.reserve(EP_KEY, victim);

        boolean result = committer.execute(victim, EP_KEY, 5000L);

        assertTrue(result, "Should return true when released within timeout");
    }

    // ==================== Timeout ====================

    @Test
    void timeout_returnsFalse() {
        DecodeAdmissionTracker tracker = new DecodeAdmissionTracker();
        PreemptRateLimiter rateLimiter = new PreemptRateLimiter(10, 50);

        // Cancel action that does NOT release the reservation
        // (simulating a stuck engine that ignores the cancel)
        List<CancelCall> cancelCalls = new ArrayList<>();
        BiConsumer<Long, CancelReasonPB> cancelAction = (reqId, reason) -> {
            cancelCalls.add(new CancelCall(reqId, reason));
            // no-op — reservation stays in tracker
        };

        RunningPreemptCommitter committer = new RunningPreemptCommitter(
                cancelAction, tracker, rateLimiter);

        DecodeReservation victim = runningRes(10L, 30, 1000);
        tracker.reserve(EP_KEY, victim);

        // Use a very short timeout to keep the test fast (30ms)
        boolean result = committer.execute(victim, EP_KEY, 30L);

        assertFalse(result, "Should return false when release times out");
        assertEquals(1, cancelCalls.size(),
                "Cancel should still be called even if release times out");
    }

    // ==================== Helpers ====================

    private static DecodeReservation runningRes(long requestId, int priority,
                                                  long kvTokens) {
        DecodeReservation r = new DecodeReservation(requestId, priority,
                10_000_000L, kvTokens, EP_KEY, requestId,
                DecodeAdmissionState.RUNNING);
        r.setRunningSinceMs(System.currentTimeMillis() - 1000);
        return r;
    }

    /** Simple record of a cancel call for verification. */
    private static final class CancelCall {
        final long requestId;
        final CancelReasonPB reason;

        CancelCall(long requestId, CancelReasonPB reason) {
            this.requestId = requestId;
            this.reason = reason;
        }
    }
}
