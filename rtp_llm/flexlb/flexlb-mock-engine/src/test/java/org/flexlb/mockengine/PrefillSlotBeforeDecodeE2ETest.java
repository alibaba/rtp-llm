package org.flexlb.mockengine;

import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

class PrefillSlotBeforeDecodeE2ETest {
    private static final int INPUT_TOKENS = 2048;
    private static final int OUTPUT_TOKENS = 8;

    @Test
    @Timeout(20)
    void waitingForPreferredPrefillSlotLeavesDecodeFreeForIndependentWork() throws Exception {
        var config = new FlexlbConfig();
        var decision = DecisionPolicyConfig.single();
        config.queueScheduler().setDecision(decision);
        var ordering = QueueOrderingConfig.priority();
        ordering.setPrefillCostOrder(true);
        config.queueScheduler().setOrdering(ordering);
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(1_000_000L);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);

        try (var h = new AutoTpmE2EHarness(63220, 2, 1, "10", 1.0,
                false, true, decision, true, config)) {
            var p0 = h.prefillEndpoint(0);
            var p1 = h.prefillEndpoint(1);
            var decode = h.decodeEndpoint(0);
            long occupant = 801;
            long waiting = 802;
            long independent = 803;
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<Long> keys = call.getArgument(0);
                return Map.of(keys.getFirst() == independent ? p1.ipPort() : p0.ipPort(), 1);
            });
            h.startAutoPump(5);

            // SINGLE/NON_BATCH returns a route to the frontend. Hold that handoff on
            // P0 so its only delivery slot stays in use; D still has one free slot.
            Response occupyingRoute = h.scheduler.submit(request(h, occupant)).get(3, TimeUnit.SECONDS);
            assertTrue(occupyingRoute.isSuccess(), occupyingRoute::getErrorMessage);
            assertEquals(1, p0.getLocallyOwnedRequestCount());
            assertEquals(1, decode.routingView().dispatchUsage().occupiedRequests());

            var pending = h.scheduler.submit(request(h, waiting));
            AutoTpmE2EHarness.await(() -> p0.queuedRequestCount() == 1, 3000,
                    "the second request waits at its preferred Prefill");
            var reservationWhileWaiting = decode.reservationHandle(waiting);
            assertFalse(pending.isDone());
            assertEquals(1, decode.routingView().dispatchUsage().occupiedRequests());

            Response otherRoute = h.scheduler.submit(request(h, independent)).get(3, TimeUnit.SECONDS);
            deliver(h, otherRoute, independent, 1);
            assertFalse(pending.isDone(), "an idle, colder P1 must not steal the P0 waiter");
            h.scheduler.cancelRequest(waiting, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(8504, pending.get(3, TimeUnit.SECONDS).getCode());
            deliver(h, occupyingRoute, occupant, 0);
            AutoTpmE2EHarness.await(() -> decode.getInflightCount() == 0
                    && p0.getLocallyOwnedRequestCount() == 0 && p0.queuedRequestCount() == 0,
                    3000, "completion and cancellation release both endpoints");
            assertEquals(1, h.prefillEngines.get(0).getAcceptedCount());
            assertEquals(1, h.prefillEngines.get(1).getAcceptedCount());
            assertEquals(2, h.decodeEngines.get(0).getCompletedCount());
            h.assertAutoPumpHealthy();
            assertNull(reservationWhileWaiting,
                    "waiting for a Prefill execution slot must not acquire a Decode reservation");
        }
    }

    private static BalanceContext request(AutoTpmE2EHarness h, long id) {
        var context = h.context(id, 50, INPUT_TOKENS, OUTPUT_TOKENS);
        context.getRequest().setBlockCacheKeys(List.of(id, id + 1000));
        context.getRequest().setCacheKeyBlockSize(1024L);
        return context;
    }

    private static void deliver(AutoTpmE2EHarness h, Response route, long id, int prefillIndex) {
        assertTrue(route.isSuccess(), route::getErrorMessage);
        var worker = h.prefillEngines.get(prefillIndex);
        var prefill = route.getServerStatus().stream().filter(s -> s.getRole() == RoleType.PREFILL)
                .findFirst().orElseThrow();
        var decode = route.getServerStatus().stream().filter(s -> s.getRole() == RoleType.DECODE)
                .findFirst().orElseThrow();
        assertEquals(worker.getGrpcPort(), prefill.getGrpcPort());
        var input = MockEngineTestSupport.inputWithDecode(id, INPUT_TOKENS, decode.getGrpcPort(), OUTPUT_TOKENS);
        MockEngineTestSupport.<EngineRpcService.GenerateOutputsPB>unary(observer ->
                worker.generateStreamCall(input, observer));
    }
}
