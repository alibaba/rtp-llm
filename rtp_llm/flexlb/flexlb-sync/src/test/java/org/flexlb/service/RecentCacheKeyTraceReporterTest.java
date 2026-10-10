package org.flexlb.service;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRequirements;
import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheHitTheoryStats;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InOrder;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RecentCacheKeyTraceReporterTest {

    @Mock
    private CacheMetricsReporter cacheMetricsReporter;

    @Test
    void should_report_request_hits_against_prior_pool() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        RequestContext firstContext = mock(RequestContext.class);
        Request firstRequest = mock(Request.class);
        when(firstContext.getRequest()).thenReturn(firstRequest);
        when(firstContext.getRequirements()).thenReturn(inputs(1L, List.of(1L, 2L, 3L), 300L, 100L));

        RequestContext secondContext = mock(RequestContext.class);
        Request secondRequest = mock(Request.class);
        when(secondContext.getRequest()).thenReturn(secondRequest);
        when(secondContext.getRequirements()).thenReturn(inputs(2L, List.of(2L, 3L, 4L), 300L, 100L));

        reporter.report(firstContext);
        reporter.report(secondContext);

        InOrder inOrder = inOrder(cacheMetricsReporter);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 300L);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 200L, 300L);
    }

    @Test
    void should_skip_window_write_and_metric_when_window_switch_is_off() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig disabledConfig = new FlexlbConfig();
        disabledConfig.getObservability().getCacheHit().getRecentKeyWindow()
                .setWriteEnabled(false);
        RequestContext skippedContext = contextWithConfig(disabledConfig);
        reporter.report(skippedContext);

        FlexlbConfig enabledConfig = new FlexlbConfig();
        RequestContext nextContext = context(enabledConfig, List.of(1L, 2L));
        reporter.report(nextContext);

        verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 1024L);
    }

    @Test
    void should_write_window_but_skip_metric_when_metric_switch_is_off() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig metricOffConfig = new FlexlbConfig();
        metricOffConfig.getObservability().getCacheHit().setMetricsEnabled(false);
        RequestContext firstContext = context(metricOffConfig, List.of(1L, 2L));
        reporter.report(firstContext);
        verify(cacheMetricsReporter, never()).reportRecentCacheKeyHitMetrics(
                org.mockito.Mockito.anyLong(),
                org.mockito.Mockito.anyLong(),
                org.mockito.Mockito.anyLong());

        FlexlbConfig enabledConfig = new FlexlbConfig();
        RequestContext secondContext = context(enabledConfig, List.of(2L, 3L));
        reporter.report(secondContext);

        verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 256L, 1024L);
    }

    @Test
    void should_record_zero_theory_hit_for_empty_cache_key_request() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig config = new FlexlbConfig();
        reporter.report(context(config, List.of(), 128L, 64L));
        reporter.report(context(config, List.of(1L), 128L, 64L));

        verify(cacheMetricsReporter, org.mockito.Mockito.times(2)).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 128L);
        verify(cacheMetricsReporter, org.mockito.Mockito.times(2)).reportTheoryCacheHitMetrics(
                org.mockito.Mockito.any(CacheHitTheoryStats.Snapshot.class));
    }

    @Test
    void should_report_theory_hit_tokens_over_input_tokens() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig config = new FlexlbConfig();
        reporter.report(context(config, List.of(1L, 2L, 3L), 1024L, 256L));
        reporter.report(context(config, List.of(2L, 3L, 4L), 1024L, 256L));

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
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig config = new FlexlbConfig();
        reporter.report(context(config, List.of(13L, 17L), 2048L, 1024L));
        reporter.report(context(config, List.of(17L, 21L), 2048L, 1024L));

        InOrder inOrder = inOrder(cacheMetricsReporter);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 2048L);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 1024L, 2048L);
    }

    @Test
    void laterRequestMutationDoesNotChangeTheRegisteredCacheInput() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        List<Long> originalKeys = new ArrayList<>(List.of(11L, 22L));
        Request request = new Request();
        request.setBlockCacheKeys(originalKeys);
        request.setSeqLen(256L);
        request.setCacheKeyBlockSize(128L);
        RequestContext first = mock(RequestContext.class);
        when(first.getRequest()).thenReturn(request);
        when(first.getRequirements()).thenReturn(inputs(17L, originalKeys, 256L, 128L));

        originalKeys.clear();
        request.setSeqLen(1L);
        request.setCacheKeyBlockSize(1L);
        reporter.report(first);
        reporter.report(context(new FlexlbConfig(), List.of(11L, 22L), 256L, 128L));

        InOrder inOrder = inOrder(cacheMetricsReporter);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(60_000L, 0L, 256L);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(60_000L, 256L, 256L);
    }

    @Test
    void overflowingHitTokenProductIsCappedByRequestInput() throws Exception {
        for (long blockSize : new long[] {Long.MAX_VALUE, 1L << 62, (1L << 62) + 1}) {
            CacheMetricsReporter metrics = mock(CacheMetricsReporter.class);
            RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
            inject(reporter, "recentCacheKeyWindow", smallWindow());
            inject(reporter, "cacheMetricsReporter", metrics);
            List<Long> keys = List.of(11L, 22L, 33L, 44L);
            reporter.report(context(new FlexlbConfig(), keys, 1024L, blockSize));
            reporter.report(context(new FlexlbConfig(), keys, 1024L, blockSize));

            verify(metrics).reportRecentCacheKeyHitMetrics(60_000L, 1024L, 1024L);
            org.mockito.ArgumentCaptor<CacheHitTheoryStats.Snapshot> captor =
                    org.mockito.ArgumentCaptor.forClass(CacheHitTheoryStats.Snapshot.class);
            verify(metrics, org.mockito.Mockito.times(2)).reportTheoryCacheHitMetrics(captor.capture());
            assertEquals(1024L, captor.getAllValues().get(1).getRequestHitCount());
        }
    }

    private static void inject(Object target, String fieldName, Object value) throws Exception {
        Field field = RecentCacheKeyTraceReporter.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static RecentCacheKeyWindow smallWindow() {
        FlexlbConfig config = new FlexlbConfig();
        config.getObservability().getCacheHit().getRecentKeyWindow()
                .setDurationMs(60_000L);
        config.getObservability().getCacheHit().getRecentKeyWindow()
                .setMaxKeyOccurrences(3_200L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return new RecentCacheKeyWindow(configService);
    }

    private static RequestContext contextWithConfig(FlexlbConfig config) {
        RequestContext requestContext = mock(RequestContext.class);
        when(requestContext.getConfig()).thenReturn(config);
        return requestContext;
    }

    private static RequestContext context(FlexlbConfig config, List<Long> cacheKeys) {
        return context(config, cacheKeys, 1024L, 256L);
    }

    private static RequestContext context(FlexlbConfig config, List<Long> cacheKeys, long seqLen, long cacheKeyBlockSize) {
        RequestContext requestContext = mock(RequestContext.class);
        Request request = mock(Request.class);
        when(requestContext.getConfig()).thenReturn(config);
        when(requestContext.getRequest()).thenReturn(request);
        when(requestContext.getRequirements()).thenReturn(inputs(0L, cacheKeys, seqLen, cacheKeyBlockSize));
        return requestContext;
    }

    private static RequestRequirements inputs(long requestId, List<Long> cacheKeys,
                                             long seqLen, long cacheKeyBlockSize) {
        return new RequestRequirements(requestId, org.flexlb.dao.SchedulingMetadata.explicit(50, Long.MAX_VALUE), seqLen,
                new DecodeResources.AdmissionCapacity(0L, 100L),
                RequestRequirements.DecodeMode.IMMEDIATE,
                DecodeCostFormula.parse("running_size"), seqLen, null, cacheKeys, cacheKeyBlockSize, true, 0);
    }
}
