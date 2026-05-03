package org.flexlb.balance.dp;

import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CacheAwareGroupSelectorTest {

    private CacheAwareService cacheAwareService;
    private CacheAwareGroupSelector selector;

    @BeforeEach
    void setUp() {
        cacheAwareService = mock(CacheAwareService.class);
        selector = new CacheAwareGroupSelector(cacheAwareService);
    }

    @Test
    void selects_group_with_best_cache_hit() {
        WorkerStatus w1 = worker("10.0.0.1", 100);
        WorkerStatus w2 = worker("10.0.0.2", 100);

        List<Long> cacheKeys = List.of(1L, 2L, 3L, 4L, 5L);
        // w1 has 2 prefix blocks matched, w2 has 4 prefix blocks matched
        when(cacheAwareService.findMatchingPrefixLength(eq("10.0.0.1:8080"), eq(cacheKeys))).thenReturn(2);
        when(cacheAwareService.findMatchingPrefixLength(eq("10.0.0.2:8080"), eq(cacheKeys))).thenReturn(4);

        List<QueuedRequest> requests = List.of(
                requestWithCacheKeys(1, 1000, cacheKeys),
                requestWithCacheKeys(2, 800, cacheKeys));

        DispatchContext ctx = new DispatchContext("m", 4, new FlexlbConfig(), requests);
        WorkerStatus selected = selector.select(List.of(w1, w2), ctx);

        // w2 has more cache hits -> lower prefill compute -> lower score -> selected
        assertEquals("10.0.0.2", selected.getIp());
    }

    @Test
    void falls_back_to_least_loaded_when_no_cache_keys() {
        WorkerStatus w1 = worker("10.0.0.1", 100, 500);
        WorkerStatus w2 = worker("10.0.0.2", 100, 100);

        List<QueuedRequest> requests = List.of(
                requestWithCacheKeys(1, 500, null),
                requestWithCacheKeys(2, 500, List.of()));

        DispatchContext ctx = new DispatchContext("m", 4, new FlexlbConfig(), requests);

        WorkerStatus selected = selector.select(List.of(w1, w2), ctx);
        assertEquals("10.0.0.2", selected.getIp(), "should pick least loaded group");

        verify(cacheAwareService, never()).findMatchingPrefixLength(any(), any());
    }

    @Test
    void prefers_less_loaded_group_when_cache_equal() {
        WorkerStatus w1 = worker("10.0.0.1", 100, 800);
        WorkerStatus w2 = worker("10.0.0.2", 100, 100);

        List<Long> cacheKeys = List.of(1L, 2L);
        when(cacheAwareService.findMatchingPrefixLength(eq("10.0.0.1:8080"), eq(cacheKeys))).thenReturn(1);
        when(cacheAwareService.findMatchingPrefixLength(eq("10.0.0.2:8080"), eq(cacheKeys))).thenReturn(1);

        List<QueuedRequest> requests = List.of(requestWithCacheKeys(1, 500, cacheKeys));

        DispatchContext ctx = new DispatchContext("m", 4, new FlexlbConfig(), requests);
        WorkerStatus selected = selector.select(List.of(w1, w2), ctx);

        assertEquals("10.0.0.2", selected.getIp(), "same cache hit, pick lower queue time");
    }

    @Test
    void heavy_load_overrides_better_cache() {
        WorkerStatus w1 = worker("10.0.0.1", 100, 5000);
        WorkerStatus w2 = worker("10.0.0.2", 100, 0);

        List<Long> cacheKeys = List.of(1L, 2L, 3L);
        when(cacheAwareService.findMatchingPrefixLength(eq("10.0.0.1:8080"), eq(cacheKeys))).thenReturn(3);
        when(cacheAwareService.findMatchingPrefixLength(eq("10.0.0.2:8080"), eq(cacheKeys))).thenReturn(0);

        List<QueuedRequest> requests = List.of(requestWithCacheKeys(1, 400, cacheKeys));

        DispatchContext ctx = new DispatchContext("m", 4, new FlexlbConfig(), requests);
        WorkerStatus selected = selector.select(List.of(w1, w2), ctx);

        assertEquals("10.0.0.2", selected.getIp(),
                "w1 has better cache but 5000ms queue load should lose to idle w2");
    }

    @Test
    void single_candidate_always_selected() {
        WorkerStatus w = worker("10.0.0.7", 100);
        List<QueuedRequest> requests = List.of(requestWithCacheKeys(1, 500, List.of(1L)));
        DispatchContext ctx = new DispatchContext("m", 4, new FlexlbConfig(), requests);

        assertEquals("10.0.0.7", selector.select(List.of(w), ctx).getIp());
    }

    @Test
    void empty_candidates_returns_null() {
        DispatchContext ctx = new DispatchContext("m", 4, new FlexlbConfig(), List.of());
        assertNull(selector.select(List.of(), ctx));
        assertNull(selector.select(null, ctx));
    }

    @Test
    void name_is_CACHE_AWARE() {
        assertEquals("CACHE_AWARE", selector.name());
    }

    // ============== helpers ==============

    private static WorkerStatus worker(String ip, long blockSize) {
        return worker(ip, blockSize, 0);
    }

    private static WorkerStatus worker(String ip, long blockSize, long queueTime) {
        WorkerStatus w = new WorkerStatus();
        w.setIp(ip);
        w.setPort(8080);
        w.setDpSize(4);
        w.setAlive(true);
        w.getRunningQueueTime().set(queueTime);
        CacheStatus cs = CacheStatus.builder().blockSize(blockSize).build();
        w.setCacheStatus(cs);
        return w;
    }

    private static QueuedRequest requestWithCacheKeys(long requestId, long seqLen, List<Long> cacheKeys) {
        BalanceContext ctx = new BalanceContext();
        Request r = new Request();
        r.setRequestId(requestId);
        r.setSeqLen(seqLen);
        r.setBlockCacheKeys(cacheKeys);
        ctx.setRequest(r);
        return QueuedRequest.of(ctx, new CompletableFuture<>());
    }
}
