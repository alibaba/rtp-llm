package org.flexlb.cache.match.localstandby;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.LocalStandbyConfig;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * Maintains a bounded index of blocks expected to reside on the Prefill selected by routing.
 * Worker generation and block size fence predictions against replacements and mixed key sizes.
 */
@Component
@Slf4j
public class LocalStandbyCacheManager {
    private final LocalStandbyConfig config;
    private final LocalStandbyCacheIndex index;
    private final ThreadPoolExecutor updates;
    private final Map<String, WorkerStatus> workers = new ConcurrentHashMap<>();

    public LocalStandbyCacheManager(ConfigService configService) {
        var matching = configService.loadBalanceConfig().getCacheMatching();
        boolean enabled = matching instanceof KvcmCacheMatchingConfig;
        config = matching instanceof KvcmCacheMatchingConfig kvcm
                ? kvcm.getLocalStandby() : new LocalStandbyConfig();
        index = new LocalStandbyCacheIndex(config.getTtlMs(), config.getMinimumTtlMs(),
                config.getTtlReductionStartRatio(), config.getMaximumEntries(), enabled);
        updates = enabled ? new ThreadPoolExecutor(1, 1, 0, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<>(config.getAsyncQueueCapacity()), task -> {
                    Thread thread = new Thread(task, "local-standby-update");
                    thread.setDaemon(true);
                    return thread;
                }) : null;
    }

    public boolean autoSwitch() {
        return config.isAutoSwitch();
    }

    public void observe(WorkerStatus worker) {
        if (updates != null && worker.isActiveGeneration()) {
            workers.put(worker.getIpPort(), worker);
        }
    }

    public void remove(WorkerStatus worker) {
        workers.remove(worker.getIpPort(), worker);
    }

    public Map<String, Integer> findMatchingEngines(List<Long> keys, RoleType role,
                                                    List<String> candidates, long blockSize) {
        Map<String, WorkerStatus> remaining = new HashMap<>();
        for (String address : candidates) {
            WorkerStatus worker = workers.get(address);
            if (worker != null && worker.getRole() == role && matchesGeneration(worker, blockSize)) {
                remaining.put(address, worker);
            }
        }
        Map<String, Integer> matches = new HashMap<>();
        long now = System.nanoTime();
        for (int block = 0; block < keys.size() && !remaining.isEmpty(); block++) {
            Map<String, Long> owners = index.getUnexpiredEnginesForBlock(keys.get(block), now);
            var iterator = remaining.entrySet().iterator();
            while (iterator.hasNext()) {
                var candidate = iterator.next();
                WorkerStatus worker = candidate.getValue();
                if (owners == null || !owners.containsKey(identity(worker))) {
                    if (matchesGeneration(worker, blockSize)) {
                        matches.put(candidate.getKey(), block);
                    }
                    iterator.remove();
                }
            }
        }
        remaining.forEach((address, worker) -> {
            if (matchesGeneration(worker, blockSize)) {
                matches.put(address, keys.size());
            }
        });
        return Map.copyOf(matches);
    }

    public void recordRoutedRequest(Request request, List<ServerStatus> selected) {
        if (updates == null || request.getBlockCacheKeys() == null
                || request.getBlockCacheKeys().isEmpty() || selected == null) {
            return;
        }
        List<Long> keys = List.copyOf(request.getBlockCacheKeys());
        long blockSize = request.getCacheKeyBlockSize();
        for (ServerStatus route : selected) {
            if (route == null || !route.isSuccess() || (route.getRole() != RoleType.PREFILL && route.getRole() != RoleType.PDFUSION)) {
                continue;
            }
            WorkerStatus worker = workers.get(route.getServerIp() + ":" + route.getHttpPort());
            if (worker == null || worker.getGenerationId() != route.getWorkerGeneration()) {
                continue;
            }
            try {
                updates.execute(() -> addRoutedBlocks(worker, blockSize, keys));
            } catch (RejectedExecutionException full) {
                log.debug("Local standby update rejected for request {}", request.getRequestId());
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        if (updates != null) {
            updates.shutdownNow();
        }
        workers.clear();
        index.shutdown();
    }

    @Scheduled(fixedDelay = 60_000)
    void refreshCapacityLimit() {
        if (updates == null) {
            return;
        }
        double blocks = 0;
        for (WorkerStatus worker : workers.values()) {
            var cache = worker.getCacheStatus();
            if (worker.isActiveGeneration() && cache != null && cache.getBlockSize() > 0) {
                blocks += Math.max(0, worker.getTotalKvCacheTokens() / cache.getBlockSize());
            }
        }
        long limit = config.getMaximumEntries();
        if (blocks > 0) {
            limit = Math.max(1, Math.min(limit, (long) (blocks * config.getCapacityMultiplier())));
        }
        index.updateMaximumEntries(limit);
    }

    private void addRoutedBlocks(WorkerStatus worker, long blockSize, List<Long> keys) {
        if (matchesGeneration(worker, blockSize)) {
            int rejected = index.addWorkerBlockMappings(identity(worker), keys);
            if (rejected > 0) {
                log.debug("Local standby rejected {} block mappings at capacity", rejected);
            }
        }
    }

    private boolean matchesGeneration(WorkerStatus worker, long blockSize) {
        var cache = worker.getCacheStatus();
        return blockSize > 0 && worker.isActiveGeneration() && workers.get(worker.getIpPort()) == worker
                && cache != null && cache.getBlockSize() == blockSize;
    }

    private static String identity(WorkerStatus worker) {
        return worker.getIpPort() + "#" + worker.getGenerationId();
    }
}
