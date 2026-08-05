package org.flexlb.service;

import org.flexlb.cache.core.ShardedRecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheHitTheoryStats;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InOrder;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RecentCacheKeyTraceReporterTest {

    @Mock
    private CacheMetricsReporter cacheMetricsReporter;

    @Test
    void should_report_request_hits_against_prior_pool() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "shardedRecentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        BalanceContext firstContext = mock(BalanceContext.class);
        Request firstRequest = mock(Request.class);
        when(firstContext.getRequest()).thenReturn(firstRequest);
        when(firstRequest.getBlockCacheKeys()).thenReturn(List.of(1L, 2L, 3L));
        when(firstRequest.getSeqLen()).thenReturn(300L);
        when(firstRequest.getCacheKeyBlockSize()).thenReturn(100L);

        BalanceContext secondContext = mock(BalanceContext.class);
        Request secondRequest = mock(Request.class);
        when(secondContext.getRequest()).thenReturn(secondRequest);
        when(secondRequest.getBlockCacheKeys()).thenReturn(List.of(2L, 3L, 4L));
        when(secondRequest.getSeqLen()).thenReturn(300L);
        when(secondRequest.getCacheKeyBlockSize()).thenReturn(100L);

        reporter.report(firstContext);
        reporter.report(secondContext);

        InOrder inOrder = inOrder(cacheMetricsReporter);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 300L);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 200L, 300L);
    }

    @Test
    void should_record_zero_theory_hit_for_empty_cache_key_request() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "shardedRecentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        reporter.report(context(List.of(), 128L, 64L));
        reporter.report(context(List.of(1L), 128L, 64L));

        verify(cacheMetricsReporter, org.mockito.Mockito.times(2)).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 128L);
        verify(cacheMetricsReporter, org.mockito.Mockito.times(2)).reportTheoryCacheHitMetrics(
                org.mockito.Mockito.any(CacheHitTheoryStats.Snapshot.class));
    }

    @Test
    void should_report_theory_hit_tokens_over_input_tokens() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "shardedRecentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        reporter.report(context(List.of(1L, 2L, 3L), 1024L, 256L));
        reporter.report(context(List.of(2L, 3L, 4L), 1024L, 256L));

        org.mockito.ArgumentCaptor<CacheHitTheoryStats.Snapshot> captor =
                org.mockito.ArgumentCaptor.forClass(CacheHitTheoryStats.Snapshot.class);
        verify(cacheMetricsReporter, org.mockito.Mockito.times(2)).reportTheoryCacheHitMetrics(captor.capture());
        CacheHitTheoryStats.Snapshot second = captor.getAllValues().get(1);
        assertEquals(512L, second.getRequestHitCount());
        assertEquals(1024L, second.getRequestTotalCount());
        assertEquals(512L, second.getAllHitCount());
        assertEquals(2048L, second.getAllTotalCount());
    }

    @Test
    void should_report_recent_hit_tokens_with_page_rr_cache_key_block_size() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "shardedRecentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        reporter.report(context(List.of(13L, 17L), 2048L, 1024L));
        reporter.report(context(List.of(17L, 21L), 2048L, 1024L));

        InOrder inOrder = inOrder(cacheMetricsReporter);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 2048L);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 1024L, 2048L);
    }

    private static void inject(Object target, String fieldName, Object value) throws Exception {
        Field field = RecentCacheKeyTraceReporter.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static ShardedRecentCacheKeyWindow smallWindow() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheHitTimeWindowMs(60_000L);
        config.setCacheHitMaxCacheKeys(3_200L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return new ShardedRecentCacheKeyWindow(configService);
    }

    private static BalanceContext context(List<Long> cacheKeys, long seqLen, long cacheKeyBlockSize) {
        BalanceContext balanceContext = mock(BalanceContext.class);
        Request request = mock(Request.class);
        when(balanceContext.getRequest()).thenReturn(request);
        when(request.getBlockCacheKeys()).thenReturn(cacheKeys);
        when(request.getSeqLen()).thenReturn(seqLen);
        when(request.getCacheKeyBlockSize()).thenReturn(cacheKeyBlockSize);
        return balanceContext;
    }
}
