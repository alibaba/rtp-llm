package org.flexlb.mockengine;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

class QueuedDecodeReservationE2ETest {
    @Test
    @Timeout(20)
    void queuedDecodeReservationDoesNotStarveAnIdlePrefill() throws Exception {
        var config = new FlexlbConfig();
        var decision = DecisionPolicyConfig.single();
        config.queueScheduler().setDecision(decision);
        var ordering = QueueOrderingConfig.priority();
        ordering.setPrefillCostOrder(true);
        config.queueScheduler().setOrdering(ordering);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(1_000_000L);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);

        try (var h = new AutoTpmE2EHarness(63210, 2, 1, "10", 1.0,
                false, true, decision, true, config)) {
            var p0 = h.prefillEndpoint(0);
            var p1 = h.prefillEndpoint(1);
            var decode = h.decodeEndpoint(0);
            long firstId = 301L;
            long secondId = 302L;
            // Disjoint prefixes keep each request on its own cache-affine Prefill.
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<Long> keys = call.getArgument(0);
                return Map.of(keys.getFirst() == firstId ? p0.ipPort() : p1.ipPort(), 1);
            });
            var constrained = h.prefillEngines.get(0);
            var normal = constrained.getFaultConfig();
            constrained.setFaultConfig(normal.toBuilder().kvPressureTokens(constrained.getTotalKvTokens()).build());
            h.startAutoPump(5L);
            AutoTpmE2EHarness.await(() -> p0.getStatus().getAvailableKvCacheTokens() == 0,
                    2000, "Master observes P0 out of KV capacity");

            var first = h.scheduler.submit(request(h, firstId));
            AutoTpmE2EHarness.await(() -> p0.queuedRequestCount() == 1,
                    2000, "first request waits for its cache-affine P0");
            assertEquals(0, decode.routingView().dispatchUsage().occupiedRequests());
            assertEquals(0, h.decodeEngines.get(0).getInflightCount());
            assertEquals(0, h.prefillEngines.get(1).getInflightCount());
            assertTrue(p1.getStatus().getAvailableKvCacheTokens() >= 2048);
            assertTrue(h.engineArrivalOrder.isEmpty());

            var second = h.scheduler.submit(request(h, secondId));
            try {
                assertTrue(second.get(3, TimeUnit.SECONDS).isSuccess());
            } catch (TimeoutException waiting) {
                // A blocked request can still succeed after P0 recovers below.
            }
            long acceptedOnIdlePrefill = h.prefillEngines.get(1).getAcceptedCount();
            assertFalse(first.isDone());
            assertEquals(1, p0.queuedRequestCount());

            // Remove only P0's pressure. Both requests must now finish on either version.
            constrained.setFaultConfig(normal);
            assertTrue(first.get(5, TimeUnit.SECONDS).isSuccess());
            assertTrue(second.get(5, TimeUnit.SECONDS).isSuccess());
            AutoTpmE2EHarness.await(() -> h.decodeEngines.get(0).getCompletedCount() == 2,
                    2000, "both requests complete after recovery");
            h.assertAutoPumpHealthy();
            assertEquals(1L, acceptedOnIdlePrefill, () -> "P1 must accept the second request while the first waits for P0. "
                    + "Arrival order after recovery: " + h.engineArrivalOrder);
        }
    }

    private static BalanceContext request(AutoTpmE2EHarness h, long id) {
        var context = h.context(id, 50, 2048, 8);
        context.getRequest().setBlockCacheKeys(List.of(id, id + 1000));
        context.getRequest().setCacheKeyBlockSize(1024L);
        return context;
    }
}
