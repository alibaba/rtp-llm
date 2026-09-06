package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Shared lifecycle primitives for scheduler contract tests. */
final class RequestLifecycleTestSupport {
    /** Seed cancellation in state-only fixtures without exposing an internal production operation. */
    static boolean recordCancellation(RequestSlot slot, CancelReason reason, String message) {
        return Boolean.TRUE.equals(ReflectionTestUtils.invokeMethod(slot, "recordCancellationLocked", reason, message));
    }

    /** Inspect a private decision in state-only fixtures without widening the production API. */
    static <T> T inspect(RequestSlot slot, String decision, Object... arguments) {
        synchronized (slot) {
            return ReflectionTestUtils.invokeMethod(slot, decision, arguments);
        }
    }

    private RequestLifecycleTestSupport() {
    }

    static BalanceContext context(FlexlbConfig config, long requestId) {
        Request request = new Request();
        request.setRequestId(Long.toString(requestId));
        request.setSeqLen(16L);
        BalanceContext context = new BalanceContext(config);
        context.setRequest(request);
        SchedulingTestConfig.configureRequiredValues(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    static void bind(
            RequestRegistry lifecycle, Registered registered) {
        try (AdmissionHandle admission =
                     lifecycle.claimAdmissionHandle(
                             registered.item().requestId(), registered.future())) {
            assertNotNull(admission);
            assertTrue(lifecycle.commitItemForPublication(
                    registered.item(), () -> true));
        }
    }

    static void bindRoute(RequestRegistry lifecycle, Registered registered) {
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered));
    }

    static PlacementResult.Status commitRoute(
            RequestRegistry lifecycle, Registered registered) {
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(
                registered.item().requestId(), registered.future())) {
            assertNotNull(admission);
            return lifecycle.commitRoute(registered.item(), () -> true);
        }
    }

    static DeliveryClaim claimRoute(RequestRegistry registry, ScheduledRequest item, BooleanSupplier handoff) {
        DeliveryClaim claim = claimRouteWithoutPrediction(registry, item, handoff);
        if (claim != null) { registry.setDeliveryPrediction(claim, emptyWork(), 30_000L); }
        return claim;
    }

    static DeliveryClaim claimBatch(RequestRegistry registry, ScheduledRequest item, long batchId, BooleanSupplier handoff) {
        DeliveryClaim claim = claimBatchWithoutPrediction(registry, item, batchId, handoff);
        if (claim != null) { registry.setDeliveryPrediction(claim, emptyWork(), 30_000L); }
        return claim;
    }

    static DeliveryClaim claimRouteWithoutPrediction(RequestRegistry registry, ScheduledRequest item, BooleanSupplier handoff) {
        var admission = org.mockito.Mockito.mock(PrefillAdmissionResources.CommittedAdmissionOwner.class);
        org.mockito.Mockito.when(admission.transferToEndpoint(item)).thenAnswer(call -> handoff.getAsBoolean());
        return registry.claimRouteDelivery(item, admission);
    }

    static DeliveryClaim claimBatchWithoutPrediction(RequestRegistry registry, ScheduledRequest item, long batchId,
            BooleanSupplier handoff) {
        var transaction = org.mockito.Mockito.mock(BatchDeliveryStrategy.BatchTransaction.class);
        org.mockito.Mockito.when(transaction.batchId()).thenReturn(batchId);
        org.mockito.Mockito.when(transaction.transferToEndpoint(item)).thenAnswer(call -> handoff.getAsBoolean());
        return registry.claimBatchDelivery(item, transaction);
    }

    private static WorkSnapshot emptyWork() {
        return new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L);
    }

    // State-only fixtures deliberately seed a phase without executing publication or timers.
    static void startRouteDelivery(RequestSlot slot) {
        assertNotNull(slot.claimDelivery(slot.activeItem(), DeliveryClaimKind.ROUTE_DECISION, 0L, () -> true));
    }

    static void startBatchDelivery(RequestSlot slot, long batchId) {
        assertNotNull(slot.claimDelivery(slot.activeItem(), DeliveryClaimKind.BATCH_ENQUEUE, batchId, () -> true));
    }

    static void markAcknowledged(RequestSlot slot) {
        ReflectionTestUtils.invokeMethod(slot, "transitionLocked", RequestState.Phase.ACKNOWLEDGED, "test acknowledgement");
    }

    static RequestSlot.RequestEffect acknowledge(RequestSlot slot, long batchId) {
        return ReflectionTestUtils.invokeMethod(slot, "acknowledgeDeliveryLocked", batchId, null);
    }

    static boolean prepareMember(RequestRegistry registry, ScheduledRequest item) {
        var transaction = org.mockito.Mockito.mock(BatchDeliveryStrategy.BatchTransaction.class);
        org.mockito.Mockito.when(transaction.append(item))
                .thenReturn(org.flexlb.balance.delivery.CapacityBoundary.Attempt.accepted(item));
        return registry.prepareBatchMember(item, transaction).accepted();
    }

    static void awaitGlobalCapacityWaiters(RequestScheduler scheduler, int expected)
            throws InterruptedException {
        Object coordinator = ReflectionTestUtils.getField(scheduler, "globalQueue");
        var lock = (ReentrantLock)
                ReflectionTestUtils.getField(coordinator, "lock");
        Object waitQueue = ReflectionTestUtils.getField(coordinator, "waitingRequests");
        var waiting = (Map<?, ?>)
                ReflectionTestUtils.getField(waitQueue, "waiting");
        // A route's close callback runs before park. Observe actual wait registration
        // under the coordinator lock, rather than treating that callback as a barrier.
        awaitCondition(() -> {
            lock.lock();
            try {
                return waiting.size() == expected;
            } finally {
                lock.unlock();
            }
        });
    }

    static void await(CountDownLatch latch) {
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                throw new AssertionError("latch was not released");
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(
                    "interrupted while awaiting latch", interrupted);
        }
    }

    static void awaitCondition(BooleanSupplier condition)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            Thread.sleep(1L);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true");
    }

    record Registered(
            ScheduledRequest item,
            CompletableFuture<Response> future) {
    }
}
