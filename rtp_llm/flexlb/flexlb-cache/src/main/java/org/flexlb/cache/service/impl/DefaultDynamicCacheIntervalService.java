package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WorkerRegistryConfig;
import org.springframework.stereotype.Service;

import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Default implementation of DynamicIntervalManager.
 * Provides thread-safe dynamic interval adjustment based on cache diff statistics.
 *
 * @author FlexLB
 */
@Service
@Slf4j
public class DefaultDynamicCacheIntervalService implements DynamicCacheIntervalService {

    private static final long DEFAULT_INTERVAL_MS = 100L;
    private static final int ROLLING_WINDOW_SIZE = 30;
    private static final double DAMPENING_FACTOR = 0.3;
    private static final double ADJUSTMENT_THRESHOLD = 0.1;

    private final int targetDiffSize;
    private final long minIntervalMs;
    private final long maxIntervalMs;
    private final AtomicLong currentIntervalMs;

    // Thread-safe state management
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    // Rolling average for diff size tracking
    private final int[] diffHistory = new int[ROLLING_WINDOW_SIZE];
    private int historyIndex;
    private int historySize;
    private double rollingAverage = 0.0;

    public DefaultDynamicCacheIntervalService(ConfigService configService) {
        WorkerRegistryConfig.CacheStatusConfig config = configService
                .loadBalanceConfig().getWorkerRegistry().getCacheStatus();
        this.targetDiffSize = config.getTargetDiffSize();
        this.minIntervalMs = config.getMinRefreshIntervalMs();
        this.maxIntervalMs = config.getMaxRefreshIntervalMs();
        this.currentIntervalMs = new AtomicLong(
                Math.max(minIntervalMs, Math.min(maxIntervalMs, DEFAULT_INTERVAL_MS)));

        log.info("DefaultDynamicIntervalManager initialized - target:{}, min:{}ms, max:{}ms, current:{}ms",
                targetDiffSize, minIntervalMs, maxIntervalMs, currentIntervalMs.get());
    }

    @Override
    public void updateDiffStatistics(int diffSize) {
        updateRollingAverage(diffSize);
        adjustIntervalIfNeeded();
    }

    @Override
    public long getCurrentIntervalMs() {
        return currentIntervalMs.get();
    }

    /**
     * Updates the rolling average with the new diff size
     */
    private void updateRollingAverage(int diffSize) {
        lock.writeLock().lock();
        try {
            diffHistory[historyIndex] = diffSize;
            historyIndex = (historyIndex + 1) % ROLLING_WINDOW_SIZE;

            if (historySize < ROLLING_WINDOW_SIZE) {
                historySize++;
            }

            // Calculate rolling average
            long sum = 0;
            for (int i = 0; i < historySize; i++) {
                sum += diffHistory[i];
            }
            rollingAverage = (double) sum / historySize;

        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Adjusts the interval based on rolling average and target diff size
     */
    private void adjustIntervalIfNeeded() {
        lock.readLock().lock();
        try {
            if (historySize < 3) {
                // Need at least 3 samples for stable adjustment
                return;
            }

            double deviation = (rollingAverage - targetDiffSize) / targetDiffSize;

            // Only adjust if deviation exceeds threshold
            if (Math.abs(deviation) < ADJUSTMENT_THRESHOLD) {
                return;
            }

            long currentInterval = currentIntervalMs.get();
            long newInterval;

            if (rollingAverage > targetDiffSize) {
                // Diff too large, decrease interval (faster sync)
                newInterval = Math.round(currentInterval * (1 - DAMPENING_FACTOR));
            } else {
                // Diff too small, increase interval (slower sync)
                newInterval = Math.round(currentInterval * (1 + DAMPENING_FACTOR));
            }

            // Apply bounds
            newInterval = Math.max(minIntervalMs, Math.min(maxIntervalMs, newInterval));

            if (newInterval != currentInterval) {
                currentIntervalMs.set(newInterval);
            }

        } finally {
            lock.readLock().unlock();
        }
    }
}
