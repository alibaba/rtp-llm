package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Resolves the model cache block size from healthy workers.
 */
@Slf4j
@Component
public class WorkerBlockSizeResolver {

    private static final long REFRESH_INTERVAL_MINUTES = 1L;

    private final AtomicLong cachedBlockSize = new AtomicLong();
    private final ScheduledExecutorService refreshExecutor;

    public WorkerBlockSizeResolver() {
        refreshExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "worker-block-size-refresher");
            thread.setDaemon(true);
            return thread;
        });
        refreshExecutor.scheduleWithFixedDelay(
                this::refresh,
                0,
                REFRESH_INTERVAL_MINUTES,
                TimeUnit.MINUTES);
    }

    public long resolve() {
        long blockSize = cachedBlockSize.get();
        if (blockSize <= 0) {
            refresh();
            blockSize = cachedBlockSize.get();
        }
        if (blockSize <= 0) {
            throw new IllegalStateException(
                    "block_size is unavailable from healthy worker cache status");
        }
        return blockSize;
    }

    void refresh() {
        try {
            refreshBlockSize();
        } catch (Exception e) {
            log.error("Failed to refresh block_size from worker cache status", e);
        }
    }

    private synchronized void refreshBlockSize() {
        ModelWorkerStatus modelWorkerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        Set<Long> detectedBlockSizes = new TreeSet<>();
        collectBlockSizes(modelWorkerStatus.getPrefillStatusMap(), detectedBlockSizes);
        collectBlockSizes(modelWorkerStatus.getDecodeStatusMap(), detectedBlockSizes);
        collectBlockSizes(modelWorkerStatus.getPdFusionStatusMap(), detectedBlockSizes);

        if (detectedBlockSizes.isEmpty()) {
            log.error("No block_size available from healthy workers; keeping cached value: {}",
                    cachedBlockSize.get());
            return;
        }
        if (detectedBlockSizes.size() > 1) {
            log.error("Inconsistent block_size values from healthy workers: {}; keeping cached value: {}",
                    detectedBlockSizes, cachedBlockSize.get());
            return;
        }

        long detectedBlockSize = detectedBlockSizes.iterator().next();
        long previousBlockSize = cachedBlockSize.getAndSet(detectedBlockSize);
        if (previousBlockSize == 0) {
            log.info("Resolved worker block_size: {}", detectedBlockSize);
        } else if (previousBlockSize != detectedBlockSize) {
            log.warn("Worker block_size changed from {} to {}", previousBlockSize, detectedBlockSize);
        }
    }

    private void collectBlockSizes(Map<String, WorkerStatus> workerStatusMap,
                                   Set<Long> blockSizes) {
        for (WorkerStatus workerStatus : workerStatusMap.values()) {
            if (workerStatus == null || !workerStatus.isAlive()) {
                continue;
            }
            CacheStatus cacheStatus = workerStatus.getCacheStatus();
            if (cacheStatus != null && cacheStatus.getBlockSize() > 0) {
                blockSizes.add(cacheStatus.getBlockSize());
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        refreshExecutor.shutdown();
    }
}
