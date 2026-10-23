package org.flexlb.cache.core;

import org.flexlb.config.ConfigService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.function.LongSupplier;

/**
 * Wrapper around a single RecentCacheKeyWindow.
 * All requests in this master share the same window, regardless of request ID.
 * The legacy class name and record signature are retained for caller compatibility.
 */
@Component
public class ShardedRecentCacheKeyWindow {
    private final RecentCacheKeyWindow window;
    private long leadershipEpoch = Long.MIN_VALUE;

    @Autowired
    public ShardedRecentCacheKeyWindow(ConfigService configService) {
        this(RecentCacheKeyWindow.resolveTimeWindowMs(configService),
                RecentCacheKeyWindow.resolveMaxCacheKeys(configService),
                System::currentTimeMillis);
    }

    ShardedRecentCacheKeyWindow(long timeWindowMs, long maxCacheKeys, LongSupplier nowSupplier) {
        this.window = new RecentCacheKeyWindow(timeWindowMs, maxCacheKeys, nowSupplier);
    }

    public synchronized RecentCacheKeyWindow.Snapshot record(long requestId, List<Long> cacheKeys) {
        return window.record(cacheKeys);
    }

    public synchronized RecentCacheKeyWindow.Snapshot recordForLeader(long epoch, List<Long> cacheKeys) {
        if (leadershipEpoch != epoch) {
            window.reset();
            leadershipEpoch = epoch;
        }
        return window.record(cacheKeys);
    }

    public synchronized void reset() {
        window.reset();
        leadershipEpoch = Long.MIN_VALUE;
    }

}
