package org.flexlb.service;

import org.flexlb.cache.core.RecentCacheKeyWindow;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class RecentCacheKeyTraceReporter {

    @Autowired(required = false)
    private RecentCacheKeyWindow recentCacheKeyWindow;

    @Autowired(required = false)
    private CacheMetricsReporter cacheMetricsReporter;

    public void report(BalanceContext balanceContext) {
        if (balanceContext == null) {
            return;
        }
        Request request = balanceContext.getRequest();
        if (request == null || recentCacheKeyWindow == null) {
            return;
        }

        RecentCacheKeyWindow.Snapshot snapshot = recentCacheKeyWindow.record(request.getBlockCacheKeys());
        if (cacheMetricsReporter == null) {
            return;
        }

        cacheMetricsReporter.reportRecentCacheKeyHitMetrics(snapshot.getTimeWindowMs(),
                snapshot.getRequestHitOccurrences(),
                snapshot.getRequestOccurrences());
    }
}
