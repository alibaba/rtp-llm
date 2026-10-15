package org.flexlb.dao.master;

import org.flexlb.dao.master.EndpointTrace.Stage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayDeque;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointTraceTest {
    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void readingExceptionDetailsDoesNotBlockProgress(boolean pollFailure) throws Exception {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        var failure = new CompletionException(new IllegalStateException("timeout"));
        var recording = new FutureTask<Void>(() -> {
            if (pollFailure) {
                poll.record(Stage.RPC_COMPLETED, failure);
            } else {
                trace.record(Stage.CACHE_CLEANUP_FAILED, failure);
            }
            return null;
        });
        Thread recorder = Thread.ofPlatform().unstarted(recording);
        try {
            // Throwable.getCause() synchronizes on the exception itself.
            synchronized (failure) {
                recorder.start();
                long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
                while (recorder.getState() != Thread.State.BLOCKED && recorder.isAlive()
                        && System.nanoTime() < deadline) {
                    Thread.sleep(1);
                }
                assertEquals(Thread.State.BLOCKED, recorder.getState());
                assertEquals(pollFailure ? "RPC_COMPLETED" : "CACHE_CLEANUP_FAILED",
                        pollFailure ? trace.snapshot().poll().stage() : trace.snapshot().phase());
                if (pollFailure) {
                    poll.record(Stage.CALLBACK_FAILED, new RejectedExecutionException("later failure"));
                    assertEquals("RPC_COMPLETED", trace.snapshot().poll().failureStage());
                    assertEquals("DETAILS_PENDING", trace.snapshot().poll().error());
                }
                poll.record(Stage.LEASE_RELEASED);
                assertTrue(trace.snapshot().poll().leaseReleased());
            }
        } finally {
            recording.get(3, TimeUnit.SECONDS);
        }
        var snapshot = trace.snapshot();
        assertEquals("IllegalStateException:timeout", pollFailure ? snapshot.poll().error() : snapshot.error());
    }

    @Test
    void failureFormattingCannotInterruptProgress() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        var failure = new IllegalStateException() {
            @Override
            public String getMessage() { throw new LinkageError("message unavailable"); }
        };
        poll.record(Stage.RPC_COMPLETED, failure);
        poll.record(EndpointTrace.Stage.CALLBACK_STARTED);
        poll.record(Stage.LEASE_RELEASED);
        assertEquals("LEASE_RELEASED", trace.snapshot().poll().stage());
        assertTrue(trace.snapshot().poll().leaseReleased());
    }

    @Test
    void distinguishesQueuedCallbackFromCallbackWaitingForWorkerLock() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 42);
        poll.record(EndpointTrace.Stage.RPC_STARTED);
        poll.record(EndpointTrace.Stage.RPC_COMPLETED);
        var queue = new ArrayDeque<Runnable>();
        CompletableFuture.completedFuture(1)
            .whenComplete((response, error) -> poll.record(Stage.CALLBACK_READY, error))
            .handleAsync((response, error) -> {
            poll.record(Stage.CALLBACK_STARTED);
            poll.record(EndpointTrace.Stage.STATUS_LOCK_ATTEMPT);
            var waiting = trace.snapshot().poll();
            assertEquals("STATUS_LOCK_ATTEMPT", waiting.stage());
            assertTrue(waiting.times().callbackStartMs() >= 0);
            assertEquals(-1, waiting.times().workerAcquiredMs());
            poll.record(Stage.CALLBACK_ENDED);
            return response;
        }, queue::add);
        var queued = trace.snapshot().poll();
        assertEquals("CALLBACK_READY", queued.stage());
        assertEquals("OK", queued.rpcResult());
        assertTrue(queued.times().rpcResultMs() >= 0);
        assertEquals(-1, queued.times().callbackStartMs());
        queue.remove().run();
        assertEquals("CALLBACK_ENDED", trace.snapshot().poll().stage());
    }

    @Test
    void completedSnapshotIsFrozenAndNextPollHasIndependentIdentity() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1);
        poll.record(Stage.RPC_COMPLETED, new IllegalStateException("timeout"));
        poll.record(Stage.LEASE_RELEASED);
        var frozen = trace.snapshot();
        trace.beginPoll("STATUS", 2).record(EndpointTrace.Stage.RPC_STARTED);
        var current = trace.snapshot();
        assertNotNull(frozen.poll());
        assertTrue(frozen.poll().leaseReleased());
        assertEquals("RPC_COMPLETED", frozen.poll().failureStage());
        assertNotEquals(frozen.poll().pollId(), current.poll().pollId());
        poll.record(EndpointTrace.Stage.RPC_COMPLETED);
        assertEquals(2, trace.snapshot().poll().syncCycleId());
        assertEquals("RPC_STARTED", trace.snapshot().poll().stage());
    }

    @Test
    void thousandsOfPollsRetainOnlyOneLatestPoll() {
        var trace = new EndpointTrace();
        for (int i = 0; i < 10_000; i++) {
            trace.beginPoll("STATUS", i).record(Stage.LEASE_RELEASED);
            trace.beginPoll("CACHE", i).record(Stage.LEASE_RELEASED);
        }
        trace.beginPoll("STATUS", 10_000);
        trace.beginPoll("CACHE", 10_000);
        assertEquals("CACHE", trace.snapshot().poll().kind());
        assertEquals(10_000, trace.snapshot().poll().syncCycleId());
    }

    @Test
    void busyAttemptsPreserveTheOwningPollProgress() {
        var worker = WorkerStatus.createDiscovered(org.flexlb.dao.route.RoleType.DECODE,
                "group", "127.0.0.1", 80, 81, "site");
        try (var lease = worker.tryBeginStatusPoll(7)) {
            lease.pollTrace().record(EndpointTrace.Stage.RPC_STARTED);
            for (int i = 0; i < 1000; i++) { assertNull(worker.tryBeginStatusPoll(8)); }
            var poll = worker.trace().snapshot().poll();
            assertEquals(7, poll.syncCycleId());
            assertEquals("RPC_STARTED", poll.stage());
            assertEquals(1000, poll.skippedBusy());
        }
    }

    @Test
    void laterCallbackFailureDoesNotHideOriginalRpcFailure() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        var failure = new IllegalStateException("deadline\nexceeded");
        poll.record(Stage.RPC_COMPLETED, new CompletionException(failure));
        poll.record(Stage.CALLBACK_READY, new CompletionException(failure));
        poll.record(Stage.CALLBACK_FAILED, new RejectedExecutionException("pool full"));
        poll.record(Stage.LEASE_RELEASED);
        var snapshot = trace.snapshot();
        assertEquals("RPC_COMPLETED", snapshot.poll().failureStage());
        assertEquals("IllegalStateException:deadline exceeded", snapshot.poll().error());
        assertEquals("ERROR", snapshot.poll().rpcResult());
        assertEquals(-1, snapshot.poll().times().convertEndMs());
        assertFalse(snapshot.compact().contains("\n"));
    }

    @Test
    void secondWorkerLockWaitDoesNotReuseTheFirstAcquisitionOrRelease() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("CACHE", 1L);
        poll.record(Stage.CACHE_LOCK_ATTEMPT);
        poll.record(Stage.CACHE_LOCK_ACQUIRED);
        poll.record(Stage.CACHE_LOCK_RELEASED);
        poll.record(Stage.CACHE_LOCK_ATTEMPT);
        var waiting = trace.snapshot().poll();
        assertEquals("CACHE_LOCK_ATTEMPT", waiting.stage());
        assertTrue(waiting.times().workerAttemptMs() >= 0L);
        assertEquals(-1L, waiting.times().workerAcquiredMs());
        assertEquals(-1L, waiting.times().workerReleasedMs());
    }

    @Test
    void snapshotRecordsCacheLockContentionAndRetirementTiming() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("CACHE", 1L);
        poll.record(Stage.CACHE_INDEX_LOCK_ATTEMPT);
        assertEquals(1, trace.snapshot().poll().indexWait().count());
        poll.record(Stage.CACHE_INDEX_LOCK_ACQUIRED);
        poll.record(Stage.CACHE_INDEX_LOCK_RELEASED);
        trace.record(Stage.GENERATION_RETIRING);
        trace.recordUnavailable("STATUS_EXPIRED", System.nanoTime() / 1000 - 11_000_000L, 10_000_000L, 0L);
        trace.record(EndpointTrace.Stage.ENDPOINT_GATE_CLOSED);
        var snapshot = trace.snapshot();
        assertTrue(snapshot.poll().indexWait().totalWaitUs() >= 0);
        assertTrue(snapshot.poll().indexWait().contendedHoldUs() >= 0);
        assertEquals("STATUS_EXPIRED", snapshot.unavailableReason());
        assertTrue(snapshot.gateClosedAtMs() >= snapshot.decidedAtMs());
        assertEquals(10_000_000, snapshot.thresholdUs());
        assertTrue(snapshot.compact().contains("gTotalUs:"));
    }

    @Test
    void concurrentCallbacksKeepIndependentMilestonesAndExactBusyCount() throws Exception {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 9);
        var start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(4)) {
            var writers = List.of(Stage.RUNNER_STARTED, Stage.RPC_COMPLETED,
                    Stage.STATUS_COMMITTED, Stage.LEASE_RELEASED).stream()
                    .map(stage -> executor.submit(() -> {
                        assertTrue(start.await(3, TimeUnit.SECONDS));
                        for (int i = 0; i < 10_000; i++) {
                            poll.record(stage);
                            poll.skippedBusy();
                        }
                        return null;
                    })).toList();
            start.countDown();
            for (int i = 0; i < 1000; i++) {
                assertEquals(9, trace.snapshot().poll().syncCycleId());
            }
            for (var writer : writers) {writer.get(5, TimeUnit.SECONDS);}
        }
        var captured = trace.snapshot().poll();
        assertEquals(40_000, captured.skippedBusy());
        assertTrue(captured.times().runnerStartMs() >= 0L);
        assertTrue(captured.times().rpcResultMs() >= 0L);
        assertTrue(captured.times().commitMs() >= 0L);
        assertTrue(captured.leaseReleased());
    }

    @Test
    void retirementPhaseIsNotOverwrittenByPollLeaseOrRepeatedGateEvents() {
        var trace = new EndpointTrace();
        trace.record(Stage.GENERATION_RETIRING);
        trace.record(Stage.ENDPOINT_GATE_CLOSED);
        long gateClosedAtMs = trace.snapshot().gateClosedAtMs();
        trace.record(Stage.CACHE_INDEX_LOCK_ATTEMPT);
        trace.record(Stage.STATUS_LEASE_LOCK_ATTEMPT);
        assertEquals("CACHE_INDEX_LOCK_ATTEMPT", trace.snapshot().phase());
        trace.record(Stage.RETIREMENT_CLEANUP_END);
        trace.record(Stage.ENDPOINT_GATE_CLOSED);
        assertEquals("RETIREMENT_CLEANUP_END", trace.snapshot().phase());
        assertEquals(gateClosedAtMs, trace.snapshot().gateClosedAtMs());
    }
}
