package org.flexlb.service;

import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheMetricsReporter;
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
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RecentCacheKeyTraceReporterTest {

    @Mock
    private CacheMetricsReporter cacheMetricsReporter;

    @Test
    void should_report_request_hits_against_prior_pool() throws Exception {
        RecentCacheKeyTraceReporter reporter = new RecentCacheKeyTraceReporter();
        inject(reporter, "recentCacheKeyWindow", new RecentCacheKeyWindow(null));
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
                RecentCacheKeyWindow.DEFAULT_TIME_WINDOW_MS, 0L, 3L);
        inOrder.verify(cacheMetricsReporter).reportRecentCacheKeyHitMetrics(
                RecentCacheKeyWindow.DEFAULT_TIME_WINDOW_MS, 2L, 3L);
    }

    private static void inject(Object target, String fieldName, Object value) throws Exception {
        Field field = RecentCacheKeyTraceReporter.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }
}
