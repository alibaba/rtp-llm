package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Real endpoint ledgers for scheduler-directory cleanup tests across package boundaries. */
public final class EndpointCleanupTestSupport {
    private EndpointCleanupTestSupport() { }

    public static void confirmDecode(DecodeEndpoint endpoint, String requestId) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(requestId);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of(requestId, running));
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    public static final class PrefillLedger {
        private final boolean batch;
        private final AtomicLong clock = new AtomicLong(100);
        private final ReentrantLock lock = new ReentrantLock();
        private final PrefillState state = new PrefillState(lock,
                PrefillActiveIndex.ordered(4, Comparator.comparing(ScheduledRequest::requestId)),
                clock::get, () -> { });
        private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });

        public PrefillLedger(boolean batch) {
            this.batch = batch;
        }

        public record Owner(ScheduledRequest item, PrefillState.Reservation reservation) { }

        public Owner commit(String requestId, long batchId, long predictedMs) {
            ScheduledRequest item = mock(ScheduledRequest.class);
            when(item.requestId()).thenReturn(requestId);
            when(item.seqLen()).thenReturn(100L);
            if (batch) {
                lock.lock();
                try {
                    assertTrue(state.enqueueActiveUnderLock(item, Long.MAX_VALUE));
                } finally {
                    lock.unlock();
                }
                try (var reservation = state.reserveBatch(item, batchId, 1,
                        generation.tryAcquireHandoff()).reservation()) {
                    assertNotNull(reservation, "the expired batch must release its sole capacity slot");
                    try (var handoff = reservation.commit(List.of(item), predictedMs)) {
                        assertNotNull(handoff);
                    }
                    return new Owner(item, reservation);
                }
            }
            try (var reservation = state.reserveUnqueuedRoute(item, predictedMs, 1).reservation()) {
                assertNotNull(reservation, "the expired individual must release its sole capacity slot");
                try (var handoff = state.commitRouteGroup(List.of(item), List.of(reservation),
                        generation.tryAcquireHandoff())) {
                    assertNotNull(handoff);
                }
                return new Owner(item, reservation);
            }
        }

        public void advanceBeyondTtl() {
            clock.addAndGet(100L);
        }

        public int sweep(Predicate<String> retain) {
            return batch ? state.evictExpiredBatches(10L, retain)
                    : state.evictExpiredIndividuals(10L, retain);
        }

        public void assertOwned(Owner owner, long predictedMs) {
            var stats = state.stats();
            assertEquals(1, stats.locallyOwnedRequests());
            assertEquals(batch ? 0 : 1, stats.individuallyOwnedRequests());
            assertEquals(batch ? 1 : 0, stats.batchCount());
            var work = state.committedSnapshot();
            assertEquals(predictedMs, work.totalRemainingWorkMs().orElseThrow());
            assertEquals(0, work.unknownRequestCount());
            if (batch) {
                assertTrue(work.requests().isEmpty());
                assertEquals(1, work.batches().size());
                assertEquals(List.of(owner.item().requestId()), work.batches().getFirst().requestIds());
                assertEquals(((PrefillState.BatchReservation) owner.reservation()).batchId(),
                        work.batches().getFirst().batchId());
                // Check the actual admission gate, not just the derived batch count.
                assertFalse(state.batchAvailability(1).isAvailable());
                assertTrue(state.batchAvailability(2).isAvailable());
            } else {
                assertTrue(work.batches().isEmpty());
                assertEquals(List.of(owner.item().requestId()), work.requests().stream()
                        .map(request -> request.requestId()).toList());
            }
        }

        public void assertStaleReleaseIsIgnored(Owner old) {
            assertFalse(state.terminalizeCommittedItem(old.item()));
            old.reservation().close();
        }

        public void assertEmpty() {
            assertEquals(0, state.stats().locallyOwnedRequests());
            assertEquals(0, state.stats().individuallyOwnedRequests());
            assertEquals(0, state.stats().batchCount());
            assertTrue(state.committedSnapshot().requests().isEmpty());
            assertTrue(state.committedSnapshot().batches().isEmpty());
            assertEquals(0L, state.committedSnapshot().totalRemainingWorkMs().orElseThrow());
            assertTrue(state.batchAvailability(1).isAvailable());
        }
    }
}
