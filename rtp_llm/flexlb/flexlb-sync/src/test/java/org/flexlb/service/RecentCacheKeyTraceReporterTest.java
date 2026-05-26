package org.flexlb.service;

import org.flexlb.cache.core.RecentCacheKeyWindow;
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

        BalanceContext firstContext = mock(BalanceContext.class);
        Request firstRequest = mock(Request.class);
        when(firstContext.getRequest()).thenReturn(firstRequest);
        when(firstRequest.getBlockCacheKeys()).thenReturn(List.of(1L, 2L, 3L));

        BalanceContext secondContext = mock(BalanceContext.class);
        Request secondRequest = mock(Request.class);
        when(secondContext.getRequest()).thenReturn(secondRequest);
        when(secondRequest.getBlockCacheKeys()).thenReturn(List.of(2L, 3L, 4L));

        reporter.report(firstContext);
        reporter.report(secondContext);

        InOrder inOrder = inOrder(cacheMetricsReporter);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 3L);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 2L, 3L);
    }

    @Test
    void should_skip_window_write_and_metric_when_window_switch_is_off() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig disabledConfig = new FlexlbConfig();
        disabledConfig.setCacheHitWindowWriteEnabled(false);
        BalanceContext skippedContext = contextWithConfig(disabledConfig);
        reporter.report(skippedContext);

        FlexlbConfig enabledConfig = new FlexlbConfig();
        BalanceContext nextContext = context(enabledConfig, List.of(1L, 2L));
        reporter.report(nextContext);

        verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 0L, 2L);
    }

    @Test
    void should_write_window_but_skip_metric_when_metric_switch_is_off() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", smallWindow());
        inject(reporter, "cacheMetricsReporter", cacheMetricsReporter);

        FlexlbConfig metricOffConfig = new FlexlbConfig();
        metricOffConfig.setCacheHitMetricReportEnabled(false);
        BalanceContext firstContext = context(metricOffConfig, List.of(1L, 2L));
        reporter.report(firstContext);
        verify(cacheMetricsReporter, never()).reportRecentCacheKeyHitMetrics(
                org.mockito.Mockito.anyLong(),
                org.mockito.Mockito.anyLong(),
                org.mockito.Mockito.anyLong());

        FlexlbConfig enabledConfig = new FlexlbConfig();
        BalanceContext secondContext = context(enabledConfig, List.of(2L, 3L));
        reporter.report(secondContext);

        verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                60_000L, 1L, 2L);
    }

    private static void inject(Object target, String fieldName, Object value) throws Exception {
        Field field = RecentCacheKeyTraceReporter.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private static RecentCacheKeyWindow smallWindow() {
        ConfigService configService = mock(ConfigService.class);
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheHitTimeWindowMs(60_000L);
        config.setCacheHitMaxCacheKeys(100L);
        when(configService.loadBalanceConfig()).thenReturn(config);
        return new RecentCacheKeyWindow(configService);
    }

    private static BalanceContext contextWithConfig(FlexlbConfig config) {
        BalanceContext balanceContext = mock(BalanceContext.class);
        when(balanceContext.getConfig()).thenReturn(config);
        return balanceContext;
    }

    private static BalanceContext context(FlexlbConfig config, List<Long> cacheKeys) {
        BalanceContext balanceContext = mock(BalanceContext.class);
        Request request = mock(Request.class);
        when(balanceContext.getConfig()).thenReturn(config);
        when(balanceContext.getRequest()).thenReturn(request);
        when(request.getBlockCacheKeys()).thenReturn(cacheKeys);
        return balanceContext;
    }
}
