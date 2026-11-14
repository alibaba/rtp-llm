package org.flexlb.mockengine;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.LongStream;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

/** Real routing and Engine feedback at 120% offered compute, with delayed transport. */
class HotOverloadAdmissionE2ETest {
    @Test
    @Timeout(50)
    void delayedFeedbackBoundsHotQueuesWithoutReducingWorkerThroughput() throws Exception {
        var config = new FlexlbConfig();
        config.getDispatcher().setType(DispatcherConfig.Type.NON_BATCH);
        config.getDispatcher().setMaxInflightPerPrefillWorker(256);
        config.queueScheduler().getDecision().setType(DecisionPolicyConfig.Type.SINGLE);
        config.setDispatchTimeoutMs(120_000L);
        config.getInternalRuntime().setRequestInactivityTimeoutMs(100_000L);
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMinPrefixHitPercent(5);
        balance.setEnableWindow(false);
        balance.setEngineQueueAdmissionThreshold(30L);
        balance.setMaxPendingComputeTokens(200_000L);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens) / 70");
        var performance = Map.<String, Object>of("test_pool_blocks", 65_536, "jitter_pct", 0.1,
                "prefill", Map.of("max_batch_requests", 32, "max_batch_tokens", 3_276_800),
                "decode", Map.of("tokens_per_step", 2, "step_base_ms", 0, "step_per_running_ms", 1));
        try (var h = new AutoTpmE2EHarness(64400, 2, 1, "sum(computeTokens) / 70", 1,
                false, false, config.queueScheduler().getDecision(), true, config, 20, performance)) {
            List<Long> prefix = LongStream.rangeClosed(1, 190).boxed().toList();
            var caches = new ArrayList<MockLruBlockCache>();
            for (var engine : h.prefillEngines) {
                var cache = (MockLruBlockCache) org.springframework.test.util.ReflectionTestUtils.getField(engine, "cache");
                assertTrue(MockEngineTestSupport.seedCache(cache, prefix));
                caches.add(cache);
            }
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                var hits = new HashMap<String, Integer>();
                for (int i = 0; i < 2; i++) { hits.put(h.prefillEndpoint(i).ipPort(), caches.get(i).prefixHitBlocks(call.getArgument(0))); }
                return hits;
            });
            h.startAutoPump(20L);
            var completed = new AtomicInteger();
            var completedByWorker = new AtomicInteger[]{new AtomicInteger(), new AtomicInteger()};
            var failure = new AtomicReference<Throwable>();
            int offered = 0;
            int maxWaiting = 0;
            // A pre-existing global backlog exposes reuse of one low queue snapshot.
            for (int i = 0; i < 124; i++) { submit(h, offered++, prefix, completed, completedByWorker, failure); }
            double qps = 2 * 70_000.0 / 5120 * 1.2;
            long start = System.nanoTime();
            int steady = 0;
            while (System.nanoTime() - start < TimeUnit.SECONDS.toNanos(20)) {
                double elapsed = (System.nanoTime() - start) / 1_000_000_000.0;
                while (steady < elapsed * qps) { submit(h, offered++, prefix, completed, completedByWorker, failure); steady++; }
                for (var engine : h.prefillEngines) {
                    maxWaiting = Math.max(maxWaiting, ((Number) engine.getSnapshot().get("waiting")).intValue());
                }
                Thread.sleep(10L);
            }
            var counts = Arrays.stream(completedByWorker).map(AtomicInteger::get).toList();
            assertNull(failure.get(), "mock transport failed");
            assertTrue(maxWaiting <= 31, "Engine queues exceed reserved headroom: " + maxWaiting + ", completions=" + counts);
            assertTrue(counts.stream().allMatch(count -> count >= 190), "both workers must keep computing: " + counts + ", snapshots=" + h.prefillEngines.stream().map(e -> {
                        var snap = e.getSnapshot();
                        return Map.of("accepted", snap.get("accepted"), "prefill_ms_count", snap.get("prefill_ms_count"), "running", snap.get("running"), "waiting", snap.get("waiting"));
                    }).toList());
            assertTrue(Math.abs(counts.get(0) - counts.get(1)) <= 0.25 * Math.max(counts.get(0), counts.get(1)),
                    "equal-cache workers must not starve each other: " + counts);
            int total = offered;
            AutoTpmE2EHarness.await(() -> completed.get() == total || failure.get() != null, 15_000L,
                    "all work must drain after overload ends");
            assertNull(failure.get());
            assertEquals(total, completed.get());
            h.assertAutoPumpHealthy();
            System.out.println("HOT_OVERLOAD_RESULT offered=" + total + " completed=" + completed.get()
                    + " max_waiting=" + maxWaiting + " per_worker_completed_at_20s=" + counts);
        }
    }

    private static void submit(AutoTpmE2EHarness h, int sequence, List<Long> prefix,
                               AtomicInteger completed, AtomicInteger[] completedByWorker, AtomicReference<Throwable> failure) {
        long id = 980_000L + sequence;
        var keys = new ArrayList<>(prefix);
        keys.addAll(LongStream.range(10_000L + sequence * 10, 10_010L + sequence * 10).boxed().toList());
        var context = h.context(id, 50, 1, 1);
        context.data().setRequest(Request.builder().requestId(id).seqLen(102_400L).maxNewTokens(1)
                .numBeams(1).model("test-model").rawPriority(50).blockCacheKeys(keys).cacheKeyBlockSize(512).build());
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 100_000L));
        h.scheduler.submit(context).thenAccept(response -> {
            if (!response.isSuccess()) { failure.compareAndSet(null, new AssertionError(response.getErrorMessage())); return; }
            int index = context.snapshot().route().prefillEp() == h.prefillEndpoint(0) ? 0 : 1;
            var input = MockEngineTestSupport.inputWithBlockKeys(id, 102_400, keys).toBuilder();
            input.getGenerateConfigBuilder().addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                    .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE).setRoleStr("DECODE")
                    .setGrpcPort(h.decodeEngines.getFirst().getGrpcPort()));
            h.engineScheduler.schedule(() -> h.prefillEngines.get(index).generateStreamCall(input.build(),
                    new io.grpc.stub.StreamObserver<>() {
                        @Override public void onNext(EngineRpcService.GenerateOutputsPB output) { }
                        @Override public void onError(Throwable error) { failure.compareAndSet(null, error); }
                        @Override public void onCompleted() { completedByWorker[index].incrementAndGet(); completed.incrementAndGet(); }
                    }), 20L, TimeUnit.MILLISECONDS);
        }).exceptionally(error -> { failure.compareAndSet(null, error); return null; });
    }
}
