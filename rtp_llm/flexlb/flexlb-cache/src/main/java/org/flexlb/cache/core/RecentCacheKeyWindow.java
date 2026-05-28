package org.flexlb.cache.core;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.LongSupplier;

/**
 * Sliding-window lookup table for request cache keys.
 *
 * <p>The hash table stores cacheKey -> occurrence count within the configured
 * time window. Expired request events decrement the corresponding count, and
 * keys are removed when their count reaches zero. Each request is checked
 * against the existing window before its keys are inserted into the window.</p>
 */
@Slf4j
@Component
public class RecentCacheKeyWindow {

    public static final long DEFAULT_TIME_WINDOW_MS = 30L * 60L * 1000L;
    static final int DEFAULT_MAX_ENTRIES = 100_000;

    private final long timeWindowMs;
    private final int maxEntries;
    private final LongSupplier nowSupplier;
    private final Deque<WindowEntry> windowEntries = new ArrayDeque<>();
    private final Map<Long, Long> cacheKeyCounts = new HashMap<>();

    private long totalOccurrences;

    @Autowired
    public RecentCacheKeyWindow(ConfigService configService) {
        this(resolveTimeWindowMs(configService), DEFAULT_MAX_ENTRIES, System::currentTimeMillis);
    }

    RecentCacheKeyWindow(long timeWindowMs, LongSupplier nowSupplier) {
        this(timeWindowMs, DEFAULT_MAX_ENTRIES, nowSupplier);
    }

    RecentCacheKeyWindow(long timeWindowMs, int maxEntries, LongSupplier nowSupplier) {
        this.timeWindowMs = normalizeTimeWindowMs(timeWindowMs);
        this.maxEntries = maxEntries > 0 ? maxEntries : DEFAULT_MAX_ENTRIES;
        this.nowSupplier = nowSupplier;
    }

    public synchronized Snapshot record(List<Long> cacheKeys) {
        return record(cacheKeys, nowSupplier.getAsLong());
    }

    synchronized Snapshot record(List<Long> cacheKeys, long nowMs) {
        evictExpired(nowMs);
        evictOverflow();
        if (cacheKeys == null || cacheKeys.isEmpty()) {
            return snapshotUnsafe(0L, 0L);
        }

        Map<Long, Long> entryCounts = new HashMap<>();
        long requestOccurrences = 0L;
        long requestHitOccurrences = 0L;
        for (Long cacheKey : cacheKeys) {
            if (cacheKey == null) {
                continue;
            }
            requestOccurrences++;
            if (cacheKeyCounts.containsKey(cacheKey)) {
                requestHitOccurrences++;
            }
            entryCounts.merge(cacheKey, 1L, Long::sum);
        }
        if (entryCounts.isEmpty()) {
            return snapshotUnsafe(0L, 0L);
        }

        windowEntries.addLast(new WindowEntry(nowMs, entryCounts));
        entryCounts.forEach((cacheKey, count) -> {
            cacheKeyCounts.merge(cacheKey, count, Long::sum);
            totalOccurrences += count;
        });
        return snapshotUnsafe(requestOccurrences, requestHitOccurrences);
    }

    public synchronized Snapshot snapshot() {
        evictExpired(nowSupplier.getAsLong());
        return snapshotUnsafe(0L, 0L);
    }

    public synchronized Snapshot clear() {
        windowEntries.clear();
        cacheKeyCounts.clear();
        totalOccurrences = 0L;
        return snapshotUnsafe(0L, 0L);
    }

    private void evictExpired(long nowMs) {
        long expireBeforeOrAt = nowMs - timeWindowMs;
        while (!windowEntries.isEmpty()) {
            WindowEntry oldest = windowEntries.peekFirst();
            if (oldest.timestampMs > expireBeforeOrAt) {
                return;
            }
            windowEntries.removeFirst();
            oldest.cacheKeyCounts.forEach(this::decrementCacheKeyCount);
        }
    }

    private void evictOverflow() {
        while (windowEntries.size() >= maxEntries) {
            WindowEntry oldest = windowEntries.removeFirst();
            oldest.cacheKeyCounts.forEach(this::decrementCacheKeyCount);
        }
    }

    private void decrementCacheKeyCount(Long cacheKey, Long expiredCount) {
        Long current = cacheKeyCounts.get(cacheKey);
        if (current == null) {
            return;
        }

        long next = current - expiredCount;
        totalOccurrences -= Math.min(current, expiredCount);
        if (next <= 0) {
            cacheKeyCounts.remove(cacheKey);
        } else {
            cacheKeyCounts.put(cacheKey, next);
        }
    }

    private Snapshot snapshotUnsafe(long requestOccurrences, long requestHitOccurrences) {
        double requestHitRatio = requestOccurrences > 0 ? requestHitOccurrences / (double) requestOccurrences : 0.0;
        return new Snapshot(timeWindowMs,
                requestOccurrences,
                requestHitOccurrences,
                requestHitRatio,
                totalOccurrences,
                cacheKeyCounts.size());
    }

    private static long resolveTimeWindowMs(ConfigService configService) {
        if (configService == null || configService.loadBalanceConfig() == null) {
            return DEFAULT_TIME_WINDOW_MS;
        }
        return configService.loadBalanceConfig().getCacheHitTimeWindowMs();
    }

    private static long normalizeTimeWindowMs(long candidateMs) {
        if (candidateMs > 0) {
            return candidateMs;
        }
        log.warn("Invalid cacheHitTimeWindowMs: {}, fallback to default: {}", candidateMs, DEFAULT_TIME_WINDOW_MS);
        return DEFAULT_TIME_WINDOW_MS;
    }

    private static class WindowEntry {
        private final long timestampMs;
        private final Map<Long, Long> cacheKeyCounts;

        private WindowEntry(long timestampMs, Map<Long, Long> cacheKeyCounts) {
            this.timestampMs = timestampMs;
            this.cacheKeyCounts = cacheKeyCounts;
        }
    }

    @Getter
    public static class Snapshot {
        private final long timeWindowMs;
        private final long requestOccurrences;
        private final long requestHitOccurrences;
        private final double requestHitRatio;
        private final long retainedOccurrences;
        private final long retainedUniqueCacheKeys;

        private Snapshot(long timeWindowMs,
                         long requestOccurrences,
                         long requestHitOccurrences,
                         double requestHitRatio,
                         long retainedOccurrences,
                         long retainedUniqueCacheKeys) {
            this.timeWindowMs = timeWindowMs;
            this.requestOccurrences = requestOccurrences;
            this.requestHitOccurrences = requestHitOccurrences;
            this.requestHitRatio = requestHitRatio;
            this.retainedOccurrences = retainedOccurrences;
            this.retainedUniqueCacheKeys = retainedUniqueCacheKeys;
        }
    }
}
