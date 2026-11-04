package org.flexlb.dao.master;

import org.flexlb.dao.master.EndpointTrace.Stage;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.net.SocketException;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.FutureTask;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointTraceTest {
    @Test
    void nestedOutOfMemoryCauseSurvivesFutureWrappers() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        var transportFailure =
                new IllegalStateException("UNKNOWN", new OutOfMemoryError("direct buffer memory"));
        var failure = new CompletionException(new ExecutionException(transportFailure));

        assertSame(transportFailure, EndpointTrace.unwrap(failure));
        poll.record(Stage.RPC_COMPLETED, failure);

        assertEquals(
                "IllegalStateException->OutOfMemoryError:UNKNOWN",
                trace.snapshot().statusPoll().error());
    }

    @Test
    void longOuterMessageCannotHideRootCauseType() {
        var trace = new EndpointTrace();
        var failure =
                new IllegalStateException(
                        "x".repeat(2000), new OutOfMemoryError("direct buffer memory"));

        trace.record(Stage.CACHE_CLEANUP_FAILED, failure);

        String summary = trace.snapshot().error();
        assertTrue(summary.startsWith("IllegalStateException->OutOfMemoryError:"));
        assertTrue(summary.length() < 300);
    }

    @Test
    void cyclicCauseChainCannotHangRecordingOrUnwrapping() {
        var first = new IllegalStateException("first");
        var second = new IllegalArgumentException("second");
        first.initCause(second);
        second.initCause(first);
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);

        assertTimeoutPreemptively(
                Duration.ofSeconds(1), () -> poll.record(Stage.RPC_COMPLETED, first));
        assertEquals(
                "IllegalStateException->IllegalArgumentException->[cycle]:first",
                trace.snapshot().statusPoll().error());
        poll.record(Stage.LEASE_RELEASED);
        assertTrue(trace.snapshot().statusPoll().leaseReleased());

        class CyclicCompletionException extends CompletionException {
            private Throwable next;

            CyclicCompletionException() {
                super((Throwable) null);
            }

            @Override
            public synchronized Throwable getCause() {
                return next;
            }
        }
        var wrapper1 = new CyclicCompletionException();
        var wrapper2 = new CyclicCompletionException();
        wrapper1.next = wrapper2;
        wrapper2.next = wrapper1;
        assertSame(
                wrapper1,
                assertTimeoutPreemptively(
                        Duration.ofSeconds(1), () -> EndpointTrace.unwrap(wrapper1)));
    }

    @Test
    void suppressedRetryFailuresRemainSeparateFromFinalCauseChain() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        var finalFailure = new IllegalStateException("deadline");
        finalFailure.addSuppressed(new SocketException("connection reset"));
        var wrapped = new ExecutionException(finalFailure);
        wrapped.addSuppressed(
                new IllegalArgumentException("previous attempt", new OutOfMemoryError("direct")));

        poll.record(Stage.RPC_COMPLETED, wrapped);
        poll.record(Stage.CALLBACK_FAILED, new RejectedExecutionException("later failure"));

        assertEquals(
                "IllegalStateException suppressed=[IllegalArgumentException->OutOfMemoryError;SocketException]:deadline",
                trace.snapshot().statusPoll().error());
        assertEquals("RPC_COMPLETED", trace.snapshot().statusPoll().failureStage());
    }

    @Test
    void longCauseAndSuppressedChainsHaveBoundedSummaries() {
        Throwable failure = new OutOfMemoryError("root");
        for (int depth = 0; depth < 100; depth++) {
            failure = new IllegalStateException("outer", failure);
        }
        for (int attempt = 0; attempt < 100; attempt++) {
            failure.addSuppressed(new SocketException("attempt " + attempt));
        }
        var trace = new EndpointTrace();

        trace.record(Stage.CACHE_CLEANUP_FAILED, failure);

        String summary = trace.snapshot().error();
        assertTrue(summary.contains("->[truncated]"));
        assertTrue(
                summary.contains(
                        "suppressed=[SocketException;SocketException;SocketException;SocketException;[truncated]]"));
        assertTrue(summary.length() < 1000);
    }

    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void readingExceptionDetailsDoesNotBlockProgress(boolean pollFailure) throws Exception {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        var failure = new CompletionException(new IllegalStateException("timeout"));
        var recording =
                new FutureTask<Void>(
                        () -> {
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
                while (recorder.getState() != Thread.State.BLOCKED
                        && recorder.isAlive()
                        && System.nanoTime() < deadline) {
                    Thread.sleep(1);
                }
                assertEquals(Thread.State.BLOCKED, recorder.getState());
                assertEquals(
                        pollFailure ? "RPC_COMPLETED" : "CACHE_CLEANUP_FAILED",
                        pollFailure
                                ? trace.snapshot().statusPoll().stage()
                                : trace.snapshot().phase());
                if (pollFailure) {
                    poll.record(
                            Stage.CALLBACK_FAILED, new RejectedExecutionException("later failure"));
                    assertEquals("RPC_COMPLETED", trace.snapshot().statusPoll().failureStage());
                    assertEquals("DETAILS_PENDING", trace.snapshot().statusPoll().error());
                }
                poll.record(Stage.LEASE_RELEASED);
                assertTrue(trace.snapshot().statusPoll().leaseReleased());
            }
        } finally {
            recording.get(3, TimeUnit.SECONDS);
        }
        var snapshot = trace.snapshot();
        assertEquals(
                "IllegalStateException:timeout",
                pollFailure ? snapshot.statusPoll().error() : snapshot.error());
    }

    @Test
    void failureFormattingCannotInterruptProgress() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 1L);
        var failure =
                new IllegalStateException() {
                    @Override
                    public String getMessage() {
                        throw new LinkageError("message unavailable");
                    }
                };
        poll.record(Stage.RPC_COMPLETED, failure);
        poll.record(EndpointTrace.Stage.CALLBACK_STARTED);
        poll.record(Stage.LEASE_RELEASED);
        assertEquals("LEASE_RELEASED", trace.snapshot().statusPoll().stage());
        assertTrue(trace.snapshot().statusPoll().leaseReleased());
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
                .handleAsync(
                        (response, error) -> {
                            poll.record(Stage.CALLBACK_STARTED);
                            poll.record(EndpointTrace.Stage.STATUS_LOCK_ATTEMPT);
                            var waiting = trace.snapshot().statusPoll();
                            assertEquals("STATUS_LOCK_ATTEMPT", waiting.stage());
                            assertTrue(waiting.times().callbackStartMs() >= 0);
                            assertEquals(-1, waiting.times().workerAcquiredMs());
                            poll.record(Stage.CALLBACK_ENDED);
                            return response;
                        },
                        queue::add);
        var queued = trace.snapshot().statusPoll();
        assertEquals("CALLBACK_READY", queued.stage());
        assertEquals("OK", queued.rpcResult());
        assertTrue(queued.times().rpcResultMs() >= 0);
        assertEquals(-1, queued.times().callbackStartMs());
        queue.remove().run();
        assertEquals("CALLBACK_ENDED", trace.snapshot().statusPoll().stage());
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
        assertNotNull(frozen.statusPoll());
        assertTrue(frozen.statusPoll().leaseReleased());
        assertEquals("RPC_COMPLETED", frozen.statusPoll().failureStage());
        assertNotEquals(frozen.statusPoll().pollId(), current.statusPoll().pollId());
        poll.record(EndpointTrace.Stage.RPC_COMPLETED);
        assertEquals(2, trace.snapshot().statusPoll().syncCycleId());
        assertEquals("RPC_STARTED", trace.snapshot().statusPoll().stage());
    }

    @Test
    void thousandsOfPollsRetainOnlyTheLatestPollOfEachKind() {
        var trace = new EndpointTrace();
        for (int i = 0; i < 10_000; i++) {
            trace.beginPoll("STATUS", i).record(Stage.LEASE_RELEASED);
            trace.beginPoll("CACHE", i).record(Stage.LEASE_RELEASED);
        }
        trace.beginPoll("STATUS", 10_000);
        trace.beginPoll("CACHE", 10_000);
        var snapshot = trace.snapshot();
        assertEquals("STATUS-20001", snapshot.statusPoll().pollId());
        assertEquals("CACHE-20002", snapshot.cachePoll().pollId());
        assertEquals(10_000, snapshot.statusPoll().syncCycleId());
        assertEquals(10_000, snapshot.cachePoll().syncCycleId());
    }

    @Test
    void completedCachePollsCannotHideTheInFlightStatusPoll() {
        var trace = new EndpointTrace();
        var status = trace.beginPoll("STATUS", 7);
        assertEquals("NOT_STARTED", trace.snapshot().statusPoll().rpcResult());
        status.recordRpcStarted(1000L);
        for (int cycle = 8; cycle < 1008; cycle++) {
            var cache = trace.beginPoll("CACHE", cycle);
            cache.record(Stage.RPC_COMPLETED);
            cache.record(Stage.CACHE_VERSION_UNCHANGED);
            cache.record(Stage.LEASE_RELEASED);
        }
        var snapshot = trace.snapshot();
        assertEquals(7, snapshot.statusPoll().syncCycleId());
        assertEquals("PENDING", snapshot.statusPoll().rpcResult());
        assertEquals("RPC_STARTED", snapshot.statusPoll().stage());
        assertFalse(snapshot.statusPoll().leaseReleased());
        assertEquals("OK", snapshot.cachePoll().rpcResult());
        assertTrue(snapshot.cachePoll().leaseReleased());
        assertTrue(snapshot.compact().contains("statusPoll={id:STATUS-1,cycle:7"));
        assertTrue(snapshot.compact().contains("cachePoll={id:CACHE-1001,cycle:1007"));

        status.record(Stage.RPC_COMPLETED);
        assertEquals("OK", trace.snapshot().statusPoll().rpcResult());
        assertEquals("PENDING", snapshot.statusPoll().rpcResult());
        trace.beginPoll("STATUS", 1008);
        status.record(Stage.LEASE_RELEASED);
        assertEquals("STATUS-1002", trace.snapshot().statusPoll().pollId());
        assertFalse(trace.snapshot().statusPoll().leaseReleased());
    }

    @Test
    void busyAttemptsPreserveTheOwningPollProgress() {
        var worker =
                WorkerStatus.createDiscovered(
                        org.flexlb.dao.route.RoleType.DECODE, "group", "127.0.0.1", 80, 81, "site");
        try (var lease = worker.tryBeginStatusPoll(7)) {
            lease.pollTrace().record(EndpointTrace.Stage.RPC_STARTED);
            for (int i = 0; i < 1000; i++) {
                assertNull(worker.tryBeginStatusPoll(8));
            }
            var poll = worker.trace().snapshot().statusPoll();
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
        assertEquals("RPC_COMPLETED", snapshot.statusPoll().failureStage());
        assertEquals("IllegalStateException:deadline exceeded", snapshot.statusPoll().error());
        assertEquals("ERROR", snapshot.statusPoll().rpcResult());
        assertEquals(-1, snapshot.statusPoll().times().convertEndMs());
        assertFalse(snapshot.compact().contains("\n"));
    }

    @Test
    void snapshotRecordsCacheLockContentionAndRetirementTiming() {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("CACHE", 1L);
        poll.record(Stage.CACHE_INDEX_LOCK_ATTEMPT);
        assertEquals(1, trace.snapshot().cachePoll().indexWait().count());
        poll.record(Stage.CACHE_INDEX_LOCK_ACQUIRED);
        poll.record(Stage.CACHE_INDEX_LOCK_RELEASED);
        trace.record(Stage.GENERATION_RETIRING);
        trace.recordUnavailable("CONSECUTIVE_WORKER_FAILURES", System.nanoTime() / 1000, 3L);
        trace.record(EndpointTrace.Stage.ENDPOINT_GATE_CLOSED);
        var snapshot = trace.snapshot();
        assertTrue(snapshot.cachePoll().indexWait().totalWaitUs() >= 0);
        assertTrue(snapshot.cachePoll().indexWait().contendedHoldUs() >= 0);
        assertEquals("CONSECUTIVE_WORKER_FAILURES", snapshot.unavailableReason());
        assertTrue(snapshot.gateClosedAtMs() >= snapshot.decidedAtMs());
        assertEquals(3L, snapshot.consecutiveFailures());
        assertTrue(snapshot.compact().contains("gTotalUs:"));
    }

    @Test
    void cacheCleanupLockIntervalsAccumulateAcrossRetriesAndFreezeBetweenThem() throws Exception {
        var trace = new EndpointTrace();
        trace.record(Stage.GENERATION_RETIRING);
        long completedWait = -1L;
        long completedHold = -1L;
        for (int interval = 0; interval < 2; interval++) {
            trace.record(Stage.CACHE_INDEX_LOCK_ATTEMPT);
            var waiting = trace.snapshot();
            long initialWait = waiting.cleanupWaitUs();
            assertTrue(initialWait >= Math.max(0L, completedWait));
            assertEquals(completedHold, waiting.cleanupHoldUs());
            Thread.sleep(2);
            var stillWaiting = trace.snapshot();
            assertTrue(stillWaiting.cleanupWaitUs() > initialWait);

            trace.record(Stage.CACHE_INDEX_LOCK_ACQUIRED);
            var acquired = trace.snapshot();
            completedWait = acquired.cleanupWaitUs();
            long initialHold = acquired.cleanupHoldUs();
            assertTrue(initialHold >= Math.max(0L, completedHold));
            Thread.sleep(2);
            var holding = trace.snapshot();
            assertEquals(completedWait, holding.cleanupWaitUs());
            assertTrue(holding.cleanupHoldUs() > initialHold);

            trace.record(Stage.CACHE_INDEX_LOCK_RELEASED);
            var released = trace.snapshot();
            completedHold = released.cleanupHoldUs();
            Thread.sleep(2);
            var later = trace.snapshot();
            assertEquals(completedWait, later.cleanupWaitUs());
            assertEquals(completedHold, later.cleanupHoldUs());
        }
    }

    @Test
    void snapshotsRemainReadableWhileCacheMutationLockIsHeldAndOtherCallersWait() throws Exception {
        var worker =
                WorkerStatus.createDiscovered(
                        org.flexlb.dao.route.RoleType.PREFILL,
                        "group",
                        "127.0.0.1",
                        80,
                        81,
                        "site");
        var trace = worker.trace();
        var statusLease = worker.tryBeginStatusPoll(1L);
        var cacheLease = worker.tryBeginCachePoll(1L);
        var waiting = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            worker.cacheMutationLock.lock();
            try {
                trace.record(Stage.GENERATION_RETIRING);
                trace.record(Stage.CACHE_INDEX_LOCK_ATTEMPT);
                trace.record(Stage.CACHE_INDEX_LOCK_ACQUIRED);
                var waiter =
                        executor.submit(
                                () -> {
                                    cacheLease
                                            .pollTrace()
                                            .record(Stage.CACHE_MUTATION_LOCK_ATTEMPT);
                                    waiting.countDown();
                                    worker.cacheMutationLock.lock();
                                    try {
                                        cacheLease
                                                .pollTrace()
                                                .record(Stage.CACHE_MUTATION_LOCK_ACQUIRED);
                                    } finally {
                                        worker.cacheMutationLock.unlock();
                                    }
                                });
                assertTrue(waiting.await(3, TimeUnit.SECONDS));
                var captured = executor.submit(trace::snapshot).get(3, TimeUnit.SECONDS);
                assertEquals("CACHE_INDEX_LOCK_ACQUIRED", captured.phase());
                assertTrue(captured.cleanupHoldUs() >= 0L);
                assertEquals(1L, captured.statusPoll().syncCycleId());
                assertEquals(1L, captured.cachePoll().syncCycleId());
                assertFalse(waiter.isDone());
            } finally {
                worker.cacheMutationLock.unlock();
                statusLease.close();
                cacheLease.close();
            }
        }
    }

    @Test
    void concurrentCallbacksKeepIndependentMilestonesAndExactBusyCount() throws Exception {
        var trace = new EndpointTrace();
        var poll = trace.beginPoll("STATUS", 9);
        var start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(4)) {
            var writers =
                    List.of(
                                    Stage.RUNNER_STARTED,
                                    Stage.RPC_COMPLETED,
                                    Stage.STATUS_COMMITTED,
                                    Stage.LEASE_RELEASED)
                            .stream()
                            .map(
                                    stage ->
                                            executor.submit(
                                                    () -> {
                                                        assertTrue(
                                                                start.await(3, TimeUnit.SECONDS));
                                                        for (int i = 0; i < 10_000; i++) {
                                                            poll.record(stage);
                                                            poll.skippedBusy();
                                                        }
                                                        return null;
                                                    }))
                            .toList();
            start.countDown();
            for (int i = 0; i < 1000; i++) {
                assertEquals(9, trace.snapshot().statusPoll().syncCycleId());
            }
            for (var writer : writers) {
                writer.get(5, TimeUnit.SECONDS);
            }
        }
        var captured = trace.snapshot().statusPoll();
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
        trace.record(Stage.RETIREMENT_CHECK_LOCK_ATTEMPT);
        assertEquals("CACHE_INDEX_LOCK_ATTEMPT", trace.snapshot().phase());
        trace.record(Stage.RETIREMENT_CLEANUP_END);
        trace.record(Stage.ENDPOINT_GATE_CLOSED);
        assertEquals("RETIREMENT_CLEANUP_END", trace.snapshot().phase());
        assertEquals(gateClosedAtMs, trace.snapshot().gateClosedAtMs());
    }
}
