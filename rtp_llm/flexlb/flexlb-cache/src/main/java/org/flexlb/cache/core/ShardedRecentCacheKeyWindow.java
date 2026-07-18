package org.flexlb.cache.core;

import org.flexlb.config.ConfigService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.function.LongSupplier;

/**
 * Sharded wrapper around RecentCacheKeyWindow to reduce lock contention.
 * Distributes records across N independent windows by requestId hash.
 * Hit rate statistics become approximate but with negligible bias under uniform distribution.
 */
@Component
public class ShardedRecentCacheKeyWindow {
    private static final int DEFAULT_SHARD_COUNT = 32;

    private final RecentCacheKeyWindow[] shards;
    private final int shardCount;

    @Autowired
    public ShardedRecentCacheKeyWindow(ConfigService configService) {
        this(DEFAULT_SHARD_COUNT,
                RecentCacheKeyWindow.resolveTimeWindowMs(configService),
                RecentCacheKeyWindow.resolveMaxCacheKeys(configService),
                System::currentTimeMillis);
    }

    public ShardedRecentCacheKeyWindow(long timeWindowMs, long maxCacheKeys) {
        this(DEFAULT_SHARD_COUNT, timeWindowMs, maxCacheKeys);
    }

    public ShardedRecentCacheKeyWindow(int shardCount, long timeWindowMs, long maxCacheKeys) {
        this(shardCount, timeWindowMs, maxCacheKeys, System::currentTimeMillis);
    }

    public ShardedRecentCacheKeyWindow(int shardCount, long timeWindowMs, long maxCacheKeys,
                                       LongSupplier nowSupplier) {
        this.shardCount = shardCount;
        this.shards = new RecentCacheKeyWindow[shardCount];
        int perShardMaxKeys = Math.max(1, (int) (maxCacheKeys / shardCount));
        for (int i = 0; i < shardCount; i++) {
            shards[i] = new RecentCacheKeyWindow(timeWindowMs, perShardMaxKeys, nowSupplier);
        }
    }

    public RecentCacheKeyWindow.Snapshot record(long requestId, List<Long> cacheKeys) {
        int idx = (int) ((requestId & Long.MAX_VALUE) % shardCount);
        return shards[idx].record(cacheKeys);
    }

    /**
     * Aggregate snapshot across all shards for approximate total hit rate.
     */
    public RecentCacheKeyWindow.Snapshot aggregateSnapshot() {
        long totalOccurrences = 0;
        long totalHitOccurrences = 0;
        long timeWindowMs = 0;
        for (RecentCacheKeyWindow shard : shards) {
            RecentCacheKeyWindow.Snapshot s = shard.snapshot();
            if (s != null) {
                totalOccurrences += s.getRequestOccurrences();
                totalHitOccurrences += s.getRequestHitOccurrences();
                timeWindowMs = s.getTimeWindowMs();
            }
        }
        return new RecentCacheKeyWindow.Snapshot(timeWindowMs, totalOccurrences, totalHitOccurrences);
    }
}
