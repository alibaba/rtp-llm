package org.flexlb.mockengine;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;

/** Regression coverage for production endpoint selection in the mock-engine loop. */
class ProductionRoutingDeadlineE2ETest {

    private static final int BASE_PORT = 63300;

    @Test
    @Timeout(10)
    void requestDeadlineIsNotMisclassifiedAsAPlacementCapacityMiss()
            throws Exception {
        DecisionPolicyConfig decision = new DecisionPolicyConfig();
        decision.setMaxRequests(2);
        decision.setMaxCollectionWaitMs(10_000L);

        // The first request's deadline is shorter than an unfilled collection
        // window. Endpoint selection must still publish it: the second request
        // immediately completes the batch, and the real mock engine can serve
        // both well before that deadline. The lifecycle coordinator, not route
        // projection, is the sole owner of terminal request deadlines.
        try (AutoTpmE2EHarness h = new AutoTpmE2EHarness(
                BASE_PORT, 1, 1, "1", 1.0,
                false, true, decision, true)) {
            h.startAutoPump(5L);

            long nearDeadlineId = 88_001L;
            long batchFollowerId = 88_002L;
            BalanceContext nearDeadline = h.context(
                    nearDeadlineId, 50, 10_000L, 8);
            nearDeadline.setSchedulingMetadata(SchedulingMetadata.explicit(
                    50, System.currentTimeMillis() + 2_000L));

            CompletableFuture<Response> first = h.scheduler.submit(nearDeadline);
            CompletableFuture<Response> second = h.scheduler.submit(
                    h.context(batchFollowerId, 50, 10_000L, 8));
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
