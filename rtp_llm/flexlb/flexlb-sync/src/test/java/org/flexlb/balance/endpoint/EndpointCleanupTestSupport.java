package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;

import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.flexlb.balance.endpoint.EndpointTestSupport.runWithStateLock;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Real endpoint ledgers for scheduler-directory cleanup tests across package boundaries.
 */
public final class EndpointCleanupTestSupport {
    private EndpointCleanupTestSupport() {}

    public static void confirmDecode(DecodeEndpoint endpoint, long requestId) {
        TaskInfo running = new TaskInfo();
        running.setRequestId(String.valueOf(requestId));
        running.setPhase(TaskPhase.KV_ALLOCATED);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(Map.of(Long.toString(requestId), running));
        response.setFinishedTaskInfo(Map.of());
        response.setAvailableKvCacheTokens(10_000L);
        response.setTotalKvCacheTokens(10_000L);
        EndpointTestSupport.applyStatus(endpoint, response).run();
    }

    /**
     * Creates committed Prefill requests with a controlled clock so cleanup tests can expire and retain exact ledger entries.
     */
    public static final class PrefillLedger {
        private final boolean batch;

        private final AtomicLong clock = new AtomicLong(100);
        private final ReentrantLock lock = new ReentrantLock();
        private final PrefillEndpoint state =
                org.flexlb.balance.endpoint.EndpointTestSupport.prefillEndpoint(
                        lock, OrderedPrefillRequestQueue.fifo(4), clock::get);
        private final EndpointRetirementTestSupport generation =
                new EndpointRetirementTestSupport(() -> {});

        public PrefillLedger(boolean batch) {
            this.batch = batch;
        }

        public Owner commit(long requestId, long batchId, long predictedMs) {
            RouteResult item = mock(RouteResult.class);
            when(item.requestId()).thenReturn(String.valueOf(requestId));
            when(item.seqLen()).thenReturn(100L);
            if (batch) {
                runWithStateLock(
                        lock,
                        () -> {
                            assertSame(
                                    item,
                                    org.flexlb.balance.endpoint.EndpointTestSupport.enqueue(
                                            state, item));
                        });
                {
                    var reservation =
                            new org.flexlb.balance.scheduler.DeliveryTestSupport.BatchInput(
                                    batchId, 1);
                    {
                        assertNotNull(
                                reservation,
                                "the expired batch must release its sole capacity slot");
                        {
                            var handoff =
                                    EndpointTestSupport.commitBatch(
                                            state, reservation, List.of(item), predictedMs);

                            assertNotNull(handoff);
                        }
                        return new Owner(item, () -> state.releaseRequest(item), batchId);
                    }
                }
            }
            {
                var reservation =
                        org.flexlb.balance.scheduler.SelectionTestSupport.registerPrefill(
                                        state, item, predictedMs, 1)
                                .value();
                try (var preparationReservation = EndpointTestSupport.preparation(reservation)) {
                    assertNotNull(
                            reservation,
                            "the expired individual must release its sole capacity slot");
                    {
                        var handoff =
                                org.flexlb.balance.endpoint.EndpointTestSupport.commitDirect(
                                        state, item);

                        assertNotNull(handoff);
                    }
                    return new Owner(item, () -> state.rollback(reservation), batchId);
                }
            }
        }

        public void advanceBeyondTtl() {
            clock.addAndGet(100L);
        }

        public int sweep(java.util.function.Predicate<String> retain) {
            return EndpointTestSupport.evictPrefill(state, 10L, retain);
        }

        public void assertOwned(Owner owner, long predictedMs) {
            var stats = state.stats();
            assertEquals(1, stats.locallyOwnedRequests());
            assertEquals(batch ? 0 : 1, stats.individuallyOwnedRequests());
            assertEquals(batch ? 1 : 0, stats.batchCount());
            var work = org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state);
            assertEquals(predictedMs, work.totalRemainingWorkMs().orElseThrow());
            assertFalse(work.hasUnknownWork());
            assertTrue(work.containsRequest(owner.item().requestId()));
            if (batch) {
                runWithStateLock(
                        lock,
                        () -> {
                            Map<?, ?> batches =
                                    (Map<?, ?>)
                                            org.springframework.test.util.ReflectionTestUtils
                                                    .getField(state, "batches");
                            assertEquals(java.util.Set.of(owner.batchId()), batches.keySet());
                        });
                // Check the actual admission gate, not just the derived batch count.
                assertFalse(state.batchCapacityAvailable(1));
                assertTrue(state.batchCapacityAvailable(2));
            }
        }

        public void assertStaleReleaseIsIgnored(Owner old) {
            assertFalse(EndpointTestSupport.releaseRequest(state, old.item()));
            old.rollback().run();
        }

        public void assertEmpty() {
            assertEquals(0, state.stats().locallyOwnedRequests());
            assertEquals(0, state.stats().individuallyOwnedRequests());
            assertEquals(0, state.stats().batchCount());
            assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
            assertFalse(
                    org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                            .hasUnknownWork());
            assertEquals(
                    0L,
                    org.flexlb.balance.endpoint.EndpointTestSupport.committedWork(state)
                            .totalRemainingWorkMs()
                            .orElseThrow());
            assertTrue(state.batchCapacityAvailable(1));
        }

        public record Owner(RouteResult item, Runnable rollback, long batchId) {}
    }
}
