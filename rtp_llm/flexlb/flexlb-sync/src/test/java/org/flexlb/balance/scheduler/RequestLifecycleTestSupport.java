package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;

import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
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
                    registered.item(), false, () -> true));
        }
    }

    static void bindRoute(
            RequestRegistry lifecycle,
            Registered registered,
            int limit,
            long acceptanceTimeoutMs) {
        assertEquals(PlacementResult.Status.SUCCESS,
                commitRoute(lifecycle, registered, limit, acceptanceTimeoutMs));
    }

    static PlacementResult.Status commitRoute(
            RequestRegistry lifecycle,
            Registered registered,
            int limit,
            long acceptanceTimeoutMs) {
        try (AdmissionMutation admission =
                     lifecycle.claimAdmissionMutation(
                             registered.item().requestId(), registered.future())) {
            assertNotNull(admission);
            return lifecycle.commitRoute(
                    registered.item(), false, limit,
                    acceptanceTimeoutMs, () -> true);
        }
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
