package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;

import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongPredicate;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

/** Real endpoint ledgers for scheduler-directory cleanup tests across package boundaries. */
public final class EndpointCleanupTestSupport {
    private EndpointCleanupTestSupport() { }

    public static void confirmDecode(DecodeEndpoint endpoint, long requestId) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(requestId);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of(Long.toString(requestId), running));
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
                PrefillActiveIndex.ordered(4, Comparator.comparingLong(RequestRoute::requestId)),
                clock::get);
        private final EndpointGenerationLifecycle generation = new EndpointGenerationLifecycle(() -> { });

        public PrefillLedger(boolean batch) {
            this.batch = batch;
        }

        public record Owner(RequestRoute item, PrefillState.Reservation reservation, long batchId) { }

        public Owner commit(long requestId, long batchId, long predictedMs) {
            RequestRoute item = mock(RequestRoute.class);
            when(item.requestId()).thenReturn(requestId);
            when(item.seqLen()).thenReturn(100L);
            if (batch) {
                lock.lock();
                try {
                    assertTrue(state.enqueueActiveLocked(item, Long.MAX_VALUE));
                } finally {
                    lock.unlock();
                }
                {
                    var reservation = state.reserveBatch(item, batchId, 1,
                        generation.tryAcquireHandoff()).reservation();
                    try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                        assertNotNull(reservation, "the expired batch must release its sole capacity slot");
                        try (var handoff = EndpointTestSupport.commitBatch(state, reservation, List.of(item), predictedMs)) {
                            assertNotNull(handoff);
                        }
                        return new Owner(item, reservation, batchId);
                    }
                }
            }
            {
                var reservation = state.reserveUnqueuedRoute(item, predictedMs, 1).reservation();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertNotNull(reservation, "the expired individual must release its sole capacity slot");
                    try (var handoff = EndpointTestSupport.commitRoutes(state, List.of(item), List.of(reservation),
                            generation.tryAcquireHandoff())) {
                        assertNotNull(handoff);
                    }
                    return new Owner(item, reservation, batchId);
                }
            }
        }

        public void advanceBeyondTtl() {
            clock.addAndGet(100L);
        }

        public int sweep(LongPredicate retain) {
            return EndpointTestSupport.evictPrefill(state, 10L, retain);
        }

        public void assertOwned(Owner owner, long predictedMs) {
            var stats = state.stats();
            assertEquals(1, stats.locallyOwnedRequests());
            assertEquals(batch ? 0 : 1, stats.individuallyOwnedRequests());
            assertEquals(batch ? 1 : 0, stats.batchCount());
            var work = state.committedSnapshot();
            assertEquals(predictedMs, work.totalRemainingWorkMs().orElseThrow());
            assertFalse(work.hasUnknownWork());
            assertTrue(work.containsRequest(owner.item().requestId()));
            if (batch) {
                lock.lock();
                try {
                    Map<?, ?> batches = (Map<?, ?>) org.springframework.test.util.ReflectionTestUtils
                            .getField(state, "batches");
                    assertEquals(java.util.Set.of(owner.batchId()), batches.keySet());
                } finally { lock.unlock(); }
                // Check the actual admission gate, not just the derived batch count.
                assertFalse(state.batchCapacityAvailable(1));
                assertTrue(state.batchCapacityAvailable(2));
            }
        }

        public void assertStaleReleaseIsIgnored(Owner old) {
            assertFalse(EndpointTestSupport.releaseRequest(state, old.item()));
            EndpointTestSupport.rollback(old.reservation());
        }

        public void assertEmpty() {
            assertEquals(0, state.stats().locallyOwnedRequests());
            assertEquals(0, state.stats().individuallyOwnedRequests());
            assertEquals(0, state.stats().batchCount());
            assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
            assertFalse(state.committedSnapshot().hasUnknownWork());
            assertEquals(0L, state.committedSnapshot().totalRemainingWorkMs().orElseThrow());
            assertTrue(state.batchCapacityAvailable(1));
        }
    }
}
