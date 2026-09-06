package org.flexlb.mockengine;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression coverage for production endpoint selection in the mock-engine
 * loop (dsv4 old-stack adaptation).
 *
 * <p>Original intake3 semantics: a request whose deadline is shorter than an
 * unfilled collection window must still be admitted by endpoint selection —
 * the lifecycle coordinator, not route projection, is the sole owner of
 * terminal request deadlines. On the dsv4 (v1) stack the fixed-window decision
 * knobs live on {@code BatchDispatcherConfig} (maxRequests / maxCollectionWaitMs)
 * and deadline ownership is enforced inside {@code PriorityScheduler}'s
 * lifecycle paths, so the equivalent assertion is: the near-deadline request
 * is neither rejected as a capacity miss nor left to expire while the batch
 * window waits; the second request completes the batch and both reach the
 * real mock engine well before the 2 s deadline.
 */
class ProductionRoutingDeadlineE2ETest {

    private static final int BASE_PORT = 63300;

    @Test
    @Timeout(10)
    void requestDeadlineIsNotMisclassifiedAsAPlacementCapacityMiss()
            throws Exception {
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT, 1, 1, "1", 1.0,
                false, true,
                dispatcher -> {
                    dispatcher.setMaxRequests(2);
                    dispatcher.setMaxCollectionWaitMs(10_000L);
                })) {
            h.startAutoPump(5L);

            long nearDeadlineId = 88_001L;
            long batchFollowerId = 88_002L;
            BalanceContext nearDeadline = h.context(
                    String.valueOf(nearDeadlineId), 50, 10_000L, 8);
            nearDeadline.setSchedulingMetadata(SchedulingMetadata.explicit(
                    50, System.currentTimeMillis() + 2_000L));

            CompletableFuture<Response> first = h.scheduler.submit(nearDeadline);
            CompletableFuture<Response> second = h.scheduler.submit(
                    h.context(String.valueOf(batchFollowerId), 50, 10_000L, 8));
            Response firstResponse = first.get(1, TimeUnit.SECONDS);
            Response secondResponse = second.get(1, TimeUnit.SECONDS);

            assertTrue(firstResponse.isSuccess(),
                    "production routing must admit serviceable work before its deadline: "
                            + firstResponse.getCode() + ": "
                            + firstResponse.getErrorMessage());
            assertTrue(secondResponse.isSuccess(),
                    "batch follower must share the successful production route");
            AutoTpmE2EHarness.await(
                    () -> h.engineArrivalOrder.contains(nearDeadlineId)
                            && h.engineArrivalOrder.contains(batchFollowerId),
                    1_000L,
                    "both admitted requests must reach the mock engine");
        }
    }
}
