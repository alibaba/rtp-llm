package org.flexlb.mockengine;

import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

class PendingPrefixPlacementTest {
    @Test
    void extendingAnAssignedPrefixWaitsForItsWorkerCacheWhileOtherRequestsProceed() throws Exception {
        var config = ConfigService.parse("""
                {"dispatchTimeoutMs":30000,"scheduler":{"decision":{"type":"SINGLE"},
                "ordering":{"type":"PRIORITY","defaultPriority":50,"prefillCostOrder":true,
                "preemption":{"allowedVictimStages":["PREFILL_QUEUED","DECODE_RESERVED","DECODE_ENGINE_OWNED"],
                "timeoutMs":1000}}},"dispatcher":{"maxInflightPerPrefillWorker":256},"router":{"roles":{
                "prefill":{"tokenBalance":{"engineQueueAdmissionThreshold":80}},
                "decode":{"availability":{"maxEngineRequests":1024}}}}}
                """);
        try (var h = new AutoTpmE2EHarness(58170, 2, 1, "sum(computeTokens) / 80", 1, true, false,
                config.queueScheduler().getDecision(), true, config, 0, Map.of("jitter_pct", 0))) {
            var index = new GlobalCacheIndex();
            when(h.cache.findMatchingEngines(any(), any(), any())).thenAnswer(call ->
                    index.batchCalculatePrefixMatchLength(call.getArgument(2), call.getArgument(0)));
            var first = LongStream.range(101, 145).boxed().toList();
            var longer = LongStream.range(101, 153).boxed().toList();
            for (int i = 0; i < 2; i++) {
                for (long key : first.subList(0, 20)) { index.addCacheBlock(key, h.prefillEndpoint(i).ipPort()); }
            }
            var parent = context(h, 401, first);
            Response assigned = h.scheduler.submit(parent).get(2, TimeUnit.SECONDS);
            assertTrue(assigned.isSuccess());
            var producer = parent.snapshot().route().prefillEp();
            var child = context(h, 402, longer);
            var waiting = h.scheduler.submit(child);
            assertThrows(TimeoutException.class, () -> waiting.get(100, TimeUnit.MILLISECONDS));
            var independent = context(h, 403, LongStream.range(201, 245).boxed().toList());
            assertTrue(h.scheduler.submit(independent).get(2, TimeUnit.SECONDS).isSuccess());
            assertFalse(waiting.isDone());
            for (long key : first.subList(0, 32)) { index.addCacheBlock(key, producer.ipPort()); }
            h.endpointRegistry.publishCacheAvailability(producer.getStatus());
            assertThrows(TimeoutException.class, () -> waiting.get(100, TimeUnit.MILLISECONDS));
            for (long key : first) { index.addCacheBlock(key, producer.ipPort()); }
            h.endpointRegistry.publishCacheAvailability(producer.getStatus());
            assertTrue(waiting.get(2, TimeUnit.SECONDS).isSuccess());
            assertSame(producer, child.snapshot().route().prefillEp());
            assertEquals(44 * 512L, child.snapshot().route().hitCache());

            // No further status or cache publication: the soft hint itself must wake the waiter,
            // re-enter normal selection and report only the actual index hit.
            var noUpdate = context(h, 404, LongStream.range(201, 253).boxed().toList());
            var fallback = h.scheduler.submit(noUpdate);
            assertThrows(TimeoutException.class, () -> fallback.get(100, TimeUnit.MILLISECONDS));
            assertTrue(fallback.get(5, TimeUnit.SECONDS).isSuccess());
            assertEquals(0L, noUpdate.snapshot().route().hitCache());
        }
    }

    private static RequestContext context(AutoTpmE2EHarness h, long id, List<Long> keys) {
        var ctx = h.context(id, 50, keys.size() * 512L, 1);
        ctx.data().setRequest(ctx.data().getRequest().toBuilder().cacheKeyBlockSize(512L).blockCacheKeys(keys).build());
        return ctx;
    }
}
