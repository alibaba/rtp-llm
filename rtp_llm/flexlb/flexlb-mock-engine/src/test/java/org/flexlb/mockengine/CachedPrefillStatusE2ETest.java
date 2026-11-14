package org.flexlb.mockengine;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

class CachedPrefillStatusE2ETest {
    @Test
    @Timeout(20)
    void cachedRunningStatusPreservesUncachedWorkUntilCompletion() throws Exception {
        var config = new FlexlbConfig();
        config.getDispatcher().setType(DispatcherConfig.Type.BATCH);
        config.getDispatcher().setMaxInflightPerPrefillWorker(256);
        config.queueScheduler().getDecision().setType(DecisionPolicyConfig.Type.SINGLE);
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setMinPrefixHitPercent(5);
        balance.setMaxPendingComputeTokens(200_000);
        balance.setEngineQueueAdmissionThreshold(30L);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens) / 70");
        var performance = Map.<String, Object>of("test_pool_blocks", 65_536, "jitter_pct", 0.1,
                "prefill", Map.of("max_batch_requests", 1, "max_batch_tokens", 102_400));
        try (var h = new AutoTpmE2EHarness(64300, 1, 1, "sum(computeTokens) / 70", 1,
                false, false, config.queueScheduler().getDecision(), true, config, 20, performance)) {
            MockLruBlockCache engineCache = (MockLruBlockCache) org.springframework.test.util.ReflectionTestUtils
                    .getField(h.prefillEngines.getFirst(), "cache");
            assertNotNull(engineCache);
            List<Long> prefix = LongStream.rangeClosed(1, 190).boxed().toList();
            assertTrue(MockEngineTestSupport.seedCache(engineCache, prefix));
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(invocation ->
                    Map.of(h.prefillEndpoint(0).getStatus().getIpPort(),
                            engineCache.prefixHitBlocks(invocation.getArgument(0))));
            h.startAutoPump(20);
            for (int i = 0; i < 3; i++) {
                long id = 950_000L + i;
                var keys = new ArrayList<>(prefix);
                keys.addAll(LongStream.range(10_000L + i * 10, 10_010L + i * 10).boxed().toList());
                var context = h.context(id, 50, 102_400, 1);
                context.data().setRequest(Request.builder().requestId(id).seqLen(102_400)
                        .maxNewTokens(1).numBeams(1).model("test-model").rawPriority(50)
                        .blockCacheKeys(keys).cacheKeyBlockSize(512).build());
                context.data().setGenerateInputPb(
                        MockEngineTestSupport.inputWithBlockKeys(id, 102_400, keys).toByteString());
                h.scheduler.submit(context);
            }
            var maxPending = new AtomicLong();
            AutoTpmE2EHarness.await(() -> {
                maxPending.accumulateAndGet(h.prefillEndpoint(0).ownership.tokenWorkload().pendingTokens(), Math::max);
                return ((Number) h.prefillEngines.getFirst().getSnapshot().get("prefill_ms_count")).longValue() >= 3;
            }, 5_000, "all cached requests complete in the real mock Engine");
            assertTrue(maxPending.get() > 0 && maxPending.get() <= 3 * 5_120,
                    "95% cache hits must survive WorkerStatus conversion: pending=" + maxPending.get());
            AutoTpmE2EHarness.await(() -> h.prefillEndpoint(0).ownership.tokenWorkload().pendingTokens() == 0,
                    2_000, "finished status releases all uncached token reservations");
            h.assertAutoPumpHealthy();
        }
    }
}
