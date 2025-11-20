package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.springframework.stereotype.Service;

import java.util.Optional;
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

    // Environment variable configuration
    private final int targetDiffSize;
    private final long minIntervalMs;
    private final long maxIntervalMs;
    
    // Rolling average configuration
    private static final int ROLLING_WINDOW_SIZE = 30;
    private static final double DAMPENING_FACTOR = 0.3;
    private static final double ADJUSTMENT_THRESHOLD = 0.1;
    
    // Thread-safe state management
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    // Rolling average for diff size tracking
    private final int[] diffHistory = new int[ROLLING_WINDOW_SIZE];
    private int historyIndex;
    private int historySize;
    private double rollingAverage = 0.0;
    
    // Statistics
    private final AtomicLong adjustmentCount = new AtomicLong(0);
    
    public DefaultDynamicCacheIntervalService() {
        
        this.targetDiffSize = Optional.ofNullable(System.getenv("CACHE_STATUS_DIFF_SIZE"))
                .map(Integer::parseInt)
                .orElse(30);
        
        this.minIntervalMs = Optional.ofNullable(System.getenv("CACHE_STATUS_MIN_INTERVAL_MS"))
                .map(Long::parseLong)
                .orElse(50L);
        
        this.maxIntervalMs = Optional.ofNullable(System.getenv("CACHE_STATUS_MAX_INTERVAL_MS"))
                .map(Long::parseLong)
                .orElse(3000L);
        
        log.info("DefaultDynamicIntervalManager initialized - target:{}, min:{}ms, max:{}ms",
            targetDiffSize, minIntervalMs, maxIntervalMs);
    }
    
    @Override
    public void updateDiffStatistics(int diffSize) {
        
        updateRollingAverage(diffSize);
        adjustIntervalIfNeeded();
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
                adjustmentCount.incrementAndGet();
            }
            
        } finally {
            lock.readLock().unlock();
        }
    }
}