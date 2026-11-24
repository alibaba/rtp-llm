package org.flexlb.mockengine;

import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithBlockKeys;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

class QueuedPrefixReuseE2ETest {
    private static final List<Long> PREFIX = List.of(11L, 12L, 13L, 14L);
    private static final int BLOCK_SIZE = 1024;
    private static final int INPUT = BLOCK_SIZE * PREFIX.size();

    @Test
    @Timeout(25)
    void queuedFollowerReusesThePrefixProducedByTheFirstRequest() throws Exception {
        verifyQueuedPrefixReuse(false);
    }

    @Test
    @Timeout(25)
    void cancelledProducerLetsFollowerUseTheRemainingResidentPrefix() throws Exception {
        verifyQueuedPrefixReuse(true);
    }

    private static void verifyQueuedPrefixReuse(boolean cancelProducer) throws Exception {
        var config = cacheAwareConfig();
        var decision = config.queueScheduler().getDecision();
        try (var h = new AutoTpmE2EHarness(63200, 2, 1, "10", 1.0, false,
                true, decision, true, config)) {
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call ->
                    prefixMatches(h, call.getArgument(0)));
            h.startAutoPump(5);
            var first = h.scheduler.submit(request(h, 101));
            AutoTpmE2EHarness.await(() -> h.prefillEndpoint(0).queuedRequestCount() + h.prefillEndpoint(1).queuedRequestCount() == 1,
                    3000, "first request placed before its dispatch window opens");
            int other = h.prefillEndpoint(0).queuedRequestCount() == 1 ? 1 : 0;
            if (cancelProducer) {
                h.prefillEndpoint(1 - other).cachePollCompleted(System.nanoTime());
            }
            // A partial copy is already available elsewhere when the follower arrives.
            warmPrefix(h, other, 99L, PREFIX.subList(0, 1));
            assertTrue(h.engineArrivalOrder.isEmpty(), "producer must still be queued");
            long initialInput = prefillTokens(h, "context_tps_with_cache");
            long initialReused = prefillTokens(h, "hit_tokens_total");
            assertEquals(BLOCK_SIZE, initialInput);
            assertEquals(0, initialReused);
            var follower = h.scheduler.submit(request(h, 102));
            if (cancelProducer) {
                AutoTpmE2EHarness.await(() -> h.scheduler.getQueuedRequestCount() == 2, 1500,
                        "both requests pending before producer cancellation");
                h.scheduler.cancelRequest(101L, 0L, CancelReason.CLIENT_CANCELLED);
                assertFalse(first.get(3, TimeUnit.SECONDS).isSuccess());
            } else {
                assertTrue(first.get(10, TimeUnit.SECONDS).isSuccess());
            }
            assertTrue(follower.get(10, TimeUnit.SECONDS).isSuccess());
            int expectedCompletions = cancelProducer ? 1 : 2;
            AutoTpmE2EHarness.await(() -> h.decodeEngines.get(0).getCompletedCount() == expectedCompletions,
                    3000, "all live requests completed");
            h.assertAutoPumpHealthy();

            long reused = prefillTokens(h, "hit_tokens_total") - initialReused;
            long input = prefillTokens(h, "context_tps_with_cache") - initialInput;
            assertEquals((long) expectedCompletions * INPUT, input);
            if (cancelProducer) {
                assertEquals(List.of(102L), h.engineArrivalOrder);
                assertEquals(BLOCK_SIZE, reused, "only the surviving one-block prefix exists");
                return;
            }
            // One cold producer + one full-prefix follower, aggregated over BOTH workers.
            assertEquals(0.5, (double) reused / input, "aggregate reused/input tokens: " + reused + "/" + input);
        }
    }

    @Test
    @Timeout(20)
    void producerCompletionWaitsForItsOwnFreshCachePollBeforeReleasingFollower() throws Exception {
        var config = cacheAwareConfig();
        var decision = DecisionPolicyConfig.single();
        config.queueScheduler().setDecision(decision);
        config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);

        try (var h = new AutoTpmE2EHarness(63260, 2, 1, "10", 1.0, false,
                true, decision, true, config)) {
            // The Engine's resident prefix and the Master's indexed view advance independently.
            var indexed = new AtomicReference<Map<String, Integer>>(Map.of());
            var followerLookup = new CountDownLatch(1);
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                var view = indexed.get();
                if (!view.isEmpty()) { followerLookup.countDown(); }
                return view;
            });
            h.startAutoPump(5);
            h.prefillEndpoint(0).cachePollCompleted(System.nanoTime());
            h.prefillEndpoint(1).cachePollCompleted(System.nanoTime());
            var producerRoute = h.scheduler.submit(request(h, 501)).get(3, TimeUnit.SECONDS);
            int producer = port(producerRoute, RoleType.PREFILL) == h.prefillEngines.get(0).getGrpcPort() ? 0 : 1;
            int peer = 1 - producer;
            warmPrefix(h, peer, 500L, PREFIX.subList(0, 1));
            indexed.set(prefixMatches(h, PREFIX));
            long staleProducerPoll = System.nanoTime();
            var follower = h.scheduler.submit(request(h, 502));
            assertTrue(followerLookup.await(3, TimeUnit.SECONDS), "follower considered the indexed one-block peer");

            generate(h, producerRoute, 501);
            h.pumpOnce();
            String producerAddress = h.prefillEndpoint(producer).ipPort();
            assertEquals(PREFIX.size(), prefixMatches(h, PREFIX).get(producerAddress));
            assertEquals(0, indexed.get().get(producerAddress));
            h.prefillEndpoint(peer).cachePollCompleted(System.nanoTime());
            h.prefillEndpoint(producer).cachePollCompleted(staleProducerPoll);
            assertThrows(TimeoutException.class, () -> follower.get(150, TimeUnit.MILLISECONDS),
                    "neither a peer poll nor a producer poll started before completion can release B");

            long pollStarted = System.nanoTime();
            indexed.set(prefixMatches(h, PREFIX));
            h.prefillEndpoint(producer).cachePollCompleted(pollStarted);
            var followerRoute = follower.get(3, TimeUnit.SECONDS);
            assertEquals(port(producerRoute, RoleType.PREFILL), port(followerRoute, RoleType.PREFILL));
            long before = prefillTokens(h, "hit_tokens_total");
            generate(h, followerRoute, 502);
            assertEquals(INPUT, prefillTokens(h, "hit_tokens_total") - before);
            h.assertAutoPumpHealthy();
        }
    }

    @ParameterizedTest(name = "follower arrives after producer completion, cache index refreshed={0}")
    @ValueSource(booleans = {false, true})
    @Timeout(20)
    void lateArrivalAfterCompletedProducerUsesIndexedPrefix(boolean refreshed) throws Exception {
        var config = cacheAwareConfig();
        var decision = DecisionPolicyConfig.single();
        config.queueScheduler().setDecision(decision);
        config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
        try (var h = new AutoTpmE2EHarness(63280, 2, 1, "10", 1.0, false,
                true, decision, true, config)) {
            var indexed = new AtomicReference<Map<String, Integer>>(Map.of());
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> indexed.get());
            h.prefillEndpoint(0).cachePollCompleted(System.nanoTime());
            h.prefillEndpoint(1).cachePollCompleted(System.nanoTime());
            var producerRoute = h.scheduler.submit(request(h, 701)).get(3, TimeUnit.SECONDS);
            int producer = port(producerRoute, RoleType.PREFILL) == h.prefillEngines.get(0).getGrpcPort() ? 0 : 1;
            int peer = 1 - producer;
            var endpoint = h.prefillEndpoint(producer);
            warmPrefix(h, peer, 700L, PREFIX.subList(0, 1));
            indexed.set(prefixMatches(h, PREFIX));
            assertEquals(1, indexed.get().get(h.prefillEndpoint(peer).ipPort()));
            assertEquals(0, indexed.get().get(endpoint.ipPort()));

            long cursor = endpoint.getStatus().appliedStatusCursor().latestFinishedTaskVersion();
            generate(h, producerRoute, 701);
            h.pumpPrefillOnce(producer);
            h.pumpDecodeOnce(0);
            assertTrue(endpoint.getStatus().appliedStatusCursor().latestFinishedTaskVersion() > cursor);
            assertEquals(0, endpoint.getLocallyOwnedRequestCount());
            var resident = prefixMatches(h, PREFIX);
            assertEquals(PREFIX.size(), resident.get(endpoint.ipPort()));
            if (refreshed) {
                long pollStarted = System.nanoTime();
                indexed.set(resident);
                endpoint.cachePollCompleted(pollStarted);
            }

            var followerRoute = h.scheduler.submit(request(h, 702)).get(3, TimeUnit.SECONDS);
            int expectedWorker = refreshed ? producer : peer;
            assertEquals(h.prefillEngines.get(expectedWorker).getGrpcPort(), port(followerRoute, RoleType.PREFILL));
            long hitsBefore = prefillTokens(h, "hit_tokens_total");
            generate(h, followerRoute, 702);
            assertEquals(refreshed ? INPUT : BLOCK_SIZE, prefillTokens(h, "hit_tokens_total") - hitsBefore);
        }
    }

    @ParameterizedTest(name = "Decode wake with {0} resident blocks on original Prefill and {1} on peer")
    @CsvSource({"1,4", "3,4", "3,3"})
    @Timeout(25)
    void decodeCapacityWakeUsesNewlyResidentLongerPrefix(int originalBlocks, int peerBlocks) throws Exception {
        var config = cacheAwareConfig();
        var decision = config.queueScheduler().getDecision();
        try (var h = new AutoTpmE2EHarness(63230, 2, 1, "10", 1.0,
                false, true, decision, true, config)) {
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call ->
                    prefixMatches(h, call.getArgument(0)));
            warmPrefix(h, 0, 201L, PREFIX.subList(0, originalBlocks));
            h.startAutoPump(5);
            var waiting = h.scheduler.submit(request(h, 202));
            var preferred = h.prefillEndpoint(0);
            AutoTpmE2EHarness.await(() -> preferred.queuedRequestCount() == 1,
                    1500, "request provisionally prefers the only cached copy");

            var decode = h.decodeEngines.get(0);
            var normal = decode.getFaultConfig();
            blockDecodeUntilWaitObserved(h, 0);
            assertFalse(waiting.isDone());

            warmPrefix(h, 1, 203L, PREFIX.subList(0, peerBlocks));
            long initialHits = prefillTokens(h, "hit_tokens_total");
            int expectedPrefill = peerBlocks > originalBlocks ? 1 : 0;
            var selectedWorker = h.prefillEngines.get(expectedPrefill);
            long initialWorkerHits = ((Number) selectedWorker.getSnapshot().get("hit_tokens_total")).longValue();
            decode.setFaultConfig(normal);
            assertEquals(selectedWorker.getGrpcPort(), port(waiting.get(8, TimeUnit.SECONDS), RoleType.PREFILL),
                    "changing Prefill requires a strictly longer resident prefix");
            AutoTpmE2EHarness.await(() -> decode.getCompletedCount() == 1,
                    3000, "waiting request completed");
            h.assertAutoPumpHealthy();
            assertEquals(List.of(202L), h.engineArrivalOrder);
            int expectedHits = Math.max(peerBlocks, originalBlocks) * BLOCK_SIZE;
            assertEquals(expectedHits, ((Number) selectedWorker.getSnapshot().get("hit_tokens_total")).longValue()
                    - initialWorkerHits);
            assertEquals(expectedHits, prefillTokens(h, "hit_tokens_total") - initialHits);
        }
    }

    @ParameterizedTest(name = "Decode waiter terminated before wake: cancel={0}")
    @ValueSource(booleans = {true, false})
    @Timeout(20)
    void aLateDecodeWakeNeverDispatchesACancelledOrExpiredRequest(boolean cancel) throws Exception {
        var config = cacheAwareConfig();
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(500);
        try (var h = new AutoTpmE2EHarness(63250, 1, 1, "10", 1.0, false,
                true, config.queueScheduler().getDecision(), true, config)) {
            h.startAutoPump(5);
            var ctx = request(h, 301);
            if (!cancel) {
                ctx.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 3000));
            }
            var pending = h.scheduler.submit(ctx);
            AutoTpmE2EHarness.await(() -> h.prefillEndpoint(0).queuedRequestCount() == 1,
                    1500, "request in collection window");
            var normal = h.decodeEngines.get(0).getFaultConfig();
            blockDecodeUntilWaitObserved(h, 0);

            if (cancel) {
                h.scheduler.cancelRequest(301L, 0L, CancelReason.CLIENT_CANCELLED);
            }
            assertFalse(pending.get(5, TimeUnit.SECONDS).isSuccess());
            h.decodeEngines.get(0).setFaultConfig(normal);
            // A new request proves the wake and the dispatch path were exercised.
            var next = h.scheduler.submit(request(h, 302)).get(4, TimeUnit.SECONDS);
            assertTrue(next.isSuccess(), next::getErrorMessage);
            AutoTpmE2EHarness.await(() -> h.decodeEngines.get(0).getCompletedCount() == 1
                            && h.decodeEndpoint(0).getInflightCount() == 0
                            && h.prefillEndpoint(0).getLocallyOwnedRequestCount() == 0,
                    3000, "new request completed and both endpoints released ownership");
            assertEquals(List.of(302L), h.engineArrivalOrder);
            h.assertAutoPumpHealthy();
        }
    }

    private static void blockDecodeUntilWaitObserved(AutoTpmE2EHarness h, int prefillIndex) throws Exception {
        var decode = h.decodeEngines.get(0);
        decode.setFaultConfig(decode.getFaultConfig().toBuilder()
                .kvPressureTokens(decode.getTotalKvTokens()).build());
        // Collection has ended and the waiting head has hit a capacity boundary.
        // P's capacity is unchanged; the only exhausted resource here is D's KV.
        AutoTpmE2EHarness.await(() -> h.prefillEndpoint(prefillIndex).captureRouteProjectionInputs()
                        .queue().admissionBlock() != null,
                5000, "the decision reached Decode capacity wait");
        assertTrue(h.engineArrivalOrder.isEmpty());
    }

    @ParameterizedTest(name = "SINGLE/NON_BATCH: {0} prefix blocks already cached on the idle worker")
    @ValueSource(ints = {1, 4})
    @Timeout(20)
    void waitForAProducerOnlyWhenItCanImproveActualReuse(int residentBlocks) throws Exception {
        var config = cacheAwareConfig();
        var decision = DecisionPolicyConfig.single();
        config.queueScheduler().setDecision(decision);
        config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);

        try (var h = new AutoTpmE2EHarness(63240, 2, 1, "sum(computeTokens) / 4 + 10", 1.0,
                false, true, decision, true, config);
             var frontend = Executors.newVirtualThreadPerTaskExecutor()) {
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call ->
                    prefixMatches(h, call.getArgument(0)));
            h.startAutoPump(5);
            var producerRoute = h.scheduler.submit(request(h, 401)).get(3, TimeUnit.SECONDS);
            int producerPort = port(producerRoute, RoleType.PREFILL);
            int producer = producerPort == h.prefillEngines.get(0).getGrpcPort() ? 0 : 1;
            int idle = 1 - producer;
            warmPrefix(h, idle, 400L, PREFIX.subList(0, residentBlocks));
            long before = prefillTokens(h, "hit_tokens_total");
            var producing = frontend.submit(() -> generate(h, producerRoute, 401));
            AutoTpmE2EHarness.await(() -> h.prefillEngines.get(producer).getRunningCount() == 1,
                    2000, "producer started computing the missing prefix");

            var followerRoute = h.scheduler.submit(request(h, 402)).get(5, TimeUnit.SECONDS);
            boolean producerStillRunning = h.prefillEngines.get(producer).getRunningCount() == 1;
            generate(h, followerRoute, 402);
            producing.get(5, TimeUnit.SECONDS);
            h.assertAutoPumpHealthy();
            assertEquals(2L, h.decodeEngines.get(0).getCompletedCount());
            assertEquals(INPUT, prefillTokens(h, "hit_tokens_total") - before,
                    "follower must reuse the full prefix with a real Decode slot available");
            if (residentBlocks == PREFIX.size()) {
                assertTrue(producerStillRunning, "the full resident copy can serve B immediately");
                assertEquals(h.prefillEngines.get(idle).getGrpcPort(), port(followerRoute, RoleType.PREFILL));
            }
        }
    }

    private static int port(Response route, RoleType role) {
        assertTrue(route.isSuccess(), route::getErrorMessage);
        return route.getServerStatus().stream().filter(status -> status.getRole() == role)
                .findFirst().orElseThrow().getGrpcPort();
    }

    private static void generate(AutoTpmE2EHarness h, Response route, long id) {
        var worker = h.services.get(port(route, RoleType.PREFILL));
        var input = inputWithBlockKeys(id, INPUT, PREFIX).toBuilder();
        input.getGenerateConfigBuilder().setMaxNewTokens(8)
                .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                        .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                        .setRoleStr("DECODE").setGrpcPort(port(route, RoleType.DECODE)));
        MockEngineTestSupport.<EngineRpcService.GenerateOutputsPB>unary(
                observer -> worker.generateStreamCall(input.build(), observer));
    }

    private static void warmPrefix(AutoTpmE2EHarness h, int index, long id, List<Long> keys) throws Exception {
        var worker = h.prefillEngines.get(index);
        long completed = worker.getCompletedCount();
        enqueue(worker, batch(id, slot(0, inputWithBlockKeys(id, keys.size() * BLOCK_SIZE, keys))));
        AutoTpmE2EHarness.await(() -> worker.getCompletedCount() == completed + 1
                        && ((Collection<?>) worker.getSnapshot().get("cache_key_set")).containsAll(keys),
                2000, "warmup completed and keys became visible");
    }

    private static FlexlbConfig cacheAwareConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.queueScheduler().setOrdering(QueueOrderingConfig.priority());
        config.getDispatcher().setPrefillCostOrder(true);
        var decision = new DecisionPolicyConfig();
        decision.setMaxRequests(16);
        decision.setMaxCollectionWaitMs(2000);
        config.queueScheduler().setDecision(decision);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        var affinity = new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(1_000_000L);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);

        return config;
    }

    private static long prefillTokens(AutoTpmE2EHarness h, String counter) {
        return h.prefillEngines.stream()
                .mapToLong(worker -> ((Number) worker.getSnapshot().get(counter)).longValue())
                .sum();
    }

    private static BalanceContext request(AutoTpmE2EHarness h, long id) {
        var context = h.context(id, 50, INPUT, 8);
        context.getRequest().setBlockCacheKeys(PREFIX);
        context.getRequest().setCacheKeyBlockSize(BLOCK_SIZE);
        var input = inputWithBlockKeys(id, INPUT, PREFIX).toBuilder();
        input.getGenerateConfigBuilder().setMaxNewTokens(8);
        context.setGenerateInputPb(input.build().toByteString());
        return context;
    }

    private static Map<String, Integer> prefixMatches(AutoTpmE2EHarness h, List<Long> keys) {
        Map<String, Integer> matches = new HashMap<>();
        for (int i = 0; i < h.prefillEngines.size(); i++) {
            Collection<?> cached = (Collection<?>) h.prefillEngines.get(i).getSnapshot().get("cache_key_set");
            int blocks = 0;
            for (long key : keys) {
                if (!cached.contains(key)) { break; }
                blocks++;
            }
            matches.put(h.prefillEndpoint(i).ipPort(), blocks);
        }
        return matches;
    }
}
