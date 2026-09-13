package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.projection.WorkSnapshot;
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

    private RequestLifecycleTestSupport() {
    }

    static BalanceContext context(FlexlbConfig config, long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(16L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        SchedulingTestConfig.configureRequiredValues(config);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    static void bind(
            RequestRegistry lifecycle, Registered registered) {
        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
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
        try (AdmissionMutation admission = lifecycle.claimAdmissionMutation(
                registered.item().requestId(), registered.future())) {
            assertNotNull(admission);
            return lifecycle.commitRoute(registered.item(), () -> true);
        }
    }

    static RequestRegistry.DeliveryClaim claimRoute(RequestRegistry lifecycle,
                                                     ScheduledRequest item,
                                                     BooleanSupplier endpointHandoff) {
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimRouteDelivery(item, endpointHandoff);
        if (claim != null) {
            lifecycle.beginDelivery(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
        }
        return claim;
    }

    static RequestRegistry.DeliveryClaim claimBatch(RequestRegistry lifecycle,
                                                     ScheduledRequest item, long batchId,
                                                     BooleanSupplier endpointHandoff) {
        RequestRegistry.DeliveryClaim claim = lifecycle.tryClaimBatchDelivery(item, batchId, endpointHandoff);
        if (claim != null) {
            lifecycle.beginDelivery(claim, new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L), 30_000L);
        }
        return claim;
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
