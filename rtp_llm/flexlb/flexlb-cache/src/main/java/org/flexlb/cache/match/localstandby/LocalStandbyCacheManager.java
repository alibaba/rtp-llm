package org.flexlb.cache.match.localstandby;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.PreDestroy;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

/** MU standby index bound to DSV's existing worker-generation and cache-mutation owner. */
@lombok.extern.slf4j.Slf4j
@Component
public final class LocalStandbyCacheManager {
    private final boolean enabled;
    private final LocalStandbyConfig config;
    private final LocalStandbyCacheIndex index;
    private final CacheMetricsReporter metrics;
    private final ThreadPoolExecutor updates;
    private final Map<String, WorkerStatus> workers = new ConcurrentHashMap<>();

    public LocalStandbyCacheManager(CacheMatchConfiguration configuration, CacheMetricsReporter metrics) {
        enabled = configuration.isLocalStandbyEnabled();
        config = enabled ? configuration.getLocalStandbyConfig() : new LocalStandbyConfig();
        this.metrics = metrics;
        updates = enabled ? new ThreadPoolExecutor(4, 10, 0, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(config.getAsyncQueueCapacity()), runnable -> {
                    Thread thread = new Thread(runnable, "local-standby-update");
                    thread.setDaemon(true); return thread;
                }, new ThreadPoolExecutor.AbortPolicy()) : null;
        index = new LocalStandbyCacheIndex(config.getTtlMs(), config.getMinimumTtlMs(),
                config.getTtlReductionStartRatio(), config.getMaximumEntries(), enabled);
    }

    /** Called under the same generation cache lock as the accepted cache snapshot. */
    public void observe(WorkerStatus worker) {
        if (!enabled || !worker.isActiveGeneration()) return;
        workers.compute(worker.getIpPort(), (address, previous) -> {
            if (previous != null && previous != worker) index.removeWorker(identity(previous));
            return worker;
        });
    }

    public void remove(WorkerStatus worker) {
        if (!enabled) return;
        workers.remove(worker.getIpPort(), worker);
        index.removeWorker(identity(worker));
    }

    public Map<String, Integer> findMatchingEngines(List<Long> keys, RoleType role,
                                                   List<String> candidates, long blockSize) {
        if (!enabled || keys.isEmpty() || blockSize <= 0) return Map.of();
        Map<String, String> remaining = new HashMap<>();
        for (String address : candidates) {
            WorkerStatus worker = workers.get(address);
            if (worker != null && worker.getRole() == role && worker.isActiveGeneration()
                    && worker.getCacheStatus() != null && worker.getCacheStatus().getBlockSize() == blockSize)
                remaining.put(address, identity(worker));
        }
        Map<String, Integer> matches = new HashMap<>();
        long now = System.nanoTime();
        for (int position = 0; position < keys.size() && !remaining.isEmpty(); position++) {
            Map<String, Long> owners = index.getUnexpiredEnginesForBlock(keys.get(position), now);
            var iterator = remaining.entrySet().iterator();
            while (iterator.hasNext()) {
                var candidate = iterator.next();
                if (owners == null || !owners.containsKey(candidate.getValue())) {
                    matches.put(candidate.getKey(), position);
                    iterator.remove();
                }
            }
        }
        remaining.keySet().forEach(address -> matches.put(address, keys.size()));
        matches.replaceAll((address, count) -> {
            WorkerStatus worker = workers.get(address);
            return worker == null ? 0 : Math.max(0, count - worker.getCacheMatchRollbackBlocks());
        });
        return Map.copyOf(matches);
    }

    /** Predictions describe routed work, never proof of computed or resident KV. */
    public void updateFromRoutedRequest(Request request, List<ServerStatus> selected) {
        if (!enabled || request == null || request.getBlockCacheKeys() == null
                || request.getBlockCacheKeys().isEmpty() || selected == null) return;
        List<Long> keys = List.copyOf(request.getBlockCacheKeys());
        List<WorkerStatus> owners = new ArrayList<>();
        for (ServerStatus status : selected) {
            if (status == null || !status.isSuccess()
                    || status.getRole() != RoleType.PREFILL && status.getRole() != RoleType.PDFUSION) continue;
            String address = status.getServerIp() + ":" + status.getHttpPort();
            WorkerStatus worker = workers.get(address);
            if (worker != null && worker.getGenerationId() == status.getWorkerGeneration()) owners.add(worker);
        }
        if (owners.isEmpty()) return;
        long blockSize = request.getCacheKeyBlockSize();
        try {
            updates.execute(() -> {
                for (WorkerStatus worker : owners) {
                    worker.cacheMutationLock.lock();
                    try {
                        if (workers.get(worker.getIpPort()) != worker || !worker.isActiveGeneration()
                                || worker.getCacheStatus() == null || worker.getCacheStatus().getBlockSize() != blockSize) continue;
                        if (index.addWorkerBlockMappings(identity(worker), keys) > 0)
                            metrics.reportLocalStandbyCapacityRejected();
                    } finally { worker.cacheMutationLock.unlock(); }
                }
            });
        } catch (RejectedExecutionException full) {
            log.warn("Local Standby update queue rejected prediction, requestId={}", request.getRequestId());
        }
    }

    @Scheduled(fixedDelay = 20000)
    public void reportCapacity() {
        if (!enabled) return;
        long blocks = 0;
        for (WorkerStatus worker : workers.values()) {
            var cache = worker.getCacheStatus();
            if (worker.isActiveGeneration() && cache != null && cache.getBlockSize() > 0)
                blocks += Math.max(0, worker.getTotalKvCacheTokens() / cache.getBlockSize());
        }
        long estimate = (long) Math.min(Long.MAX_VALUE, blocks * config.getCapacityMultiplier());
        index.updateMaximumEntries(estimate > 0 ? Math.min(config.getMaximumEntries(), estimate) : config.getMaximumEntries());
        metrics.reportLocalStandbyMappingCount(index.mappingCount());
    }

    private static String identity(WorkerStatus worker) {
        return worker.getIpPort() + "@0#" + worker.getGenerationId();
    }

    @PreDestroy public void shutdown() {
        if (updates != null) {
            updates.shutdownNow();
            try { updates.awaitTermination(1, TimeUnit.SECONDS); }
            catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        }
        index.shutdown(); workers.clear();
    }
}
