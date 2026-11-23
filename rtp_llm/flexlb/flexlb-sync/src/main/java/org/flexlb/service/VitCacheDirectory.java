package org.flexlb.service;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalCacheStatusPB;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Refreshes multimodal cache residency for live worker generations and provides group-scoped
 * candidates from immutable views. Background refresh removes expired residency snapshots.
 */
@Component
public class VitCacheDirectory {
    // Cache residency changes after each image; ten-minute snapshots lose affinity.
    static final long SYNC_INTERVAL_MS = 30_000;

    private final WorkerDirectory workers;
    private final EngineGrpcClient grpc;
    private final LBStatusConsistencyService consistency;
    private final Map<String, Long> attempts = new HashMap<>();
    private final ScheduledExecutorService sync =
            Executors.newSingleThreadScheduledExecutor(
                    r -> {
                        Thread thread = new Thread(r, "vit-cache-sync");
                        thread.setDaemon(true);
                        return thread;
                    });

    private volatile Map<String, Snapshot> snapshots = Map.of();
    private boolean wasMaster;

    public VitCacheDirectory(
            WorkerDirectory workers,
            EngineGrpcClient grpc,
            LBStatusConsistencyService consistency) {
        this.workers = workers;
        this.grpc = grpc;
        this.consistency = consistency;
    }

    @PostConstruct
    public void start() {
        sync.scheduleWithFixedDelay(this::refresh, 0, 5, TimeUnit.SECONDS);
    }

    @PreDestroy
    public void stop() {
        sync.shutdownNow();
    }

    /** Cache matches share immutable key sets from one directory view. */
    public Map<String, Candidate> candidates(String group, List<String> keys) {
        var live = workers.vitRoutingSnapshot();
        var publishedSnapshots = snapshots;
        long now = System.currentTimeMillis();
        long nowUs = System.nanoTime() / 1000;
        Map<String, Candidate> matches = new HashMap<>();
        for (WorkerStatus worker : live.values()) {
            if (!isRoutable(worker, nowUs)
                    || (group != null && !Objects.equals(group, worker.getGroup()))) {
                continue;
            }
            String address = worker.getLogicalIpPort();
            Snapshot snapshot = publishedSnapshots.get(address);
            if (snapshot != null && !snapshot.isCurrent(worker, now)) {
                snapshot = null;
            }
            int hashHits = 0;
            int gpuHits = 0;
            int cpuHits = 0;
            if (snapshot != null) {
                for (String key : keys) {
                    if (snapshot.keys().contains(key)) {
                        hashHits++;
                    }
                    if (snapshot.gpuEmbeddingKeys().contains(key)) {
                        gpuHits++;
                    } else if (snapshot.cpuEmbeddingKeys().contains(key)) {
                        cpuHits++;
                    }
                }
            }
            matches.put(
                    address,
                    new Candidate(
                            worker,
                            snapshot == null ? null : snapshot.instance(),
                            snapshot == null ? Set.of() : snapshot.keys(),
                            hashHits,
                            gpuHits + cpuHits,
                            gpuHits));
        }
        return Map.copyOf(matches);
    }

    void refresh() {
        try {
            boolean leader = !consistency.isNeedConsistency() || consistency.isMaster();
            Map<String, WorkerStatus> live = workers.vitRoutingSnapshot();
            List<WorkerStatus> due = new ArrayList<>();
            long now = System.currentTimeMillis();
            long nowUs = System.nanoTime() / 1000;
            synchronized (this) {
                prune(live, now);
                if (!leader) {
                    wasMaster = false;
                    return;
                }
                if (!wasMaster) {
                    attempts.clear();
                }
                for (WorkerStatus worker : live.values()) {
                    if (isRoutable(worker, nowUs)
                            && now - attempts.getOrDefault(worker.getLogicalIpPort(), 0L)
                                    >= SYNC_INTERVAL_MS) {
                        attempts.put(worker.getLogicalIpPort(), now);
                        due.add(worker);
                    }
                }
                wasMaster = true;
            }
            CacheVersionPB request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
            Flux.fromIterable(due)
                    .flatMap(
                            worker ->
                                    Mono.defer(
                                                    () ->
                                                            Mono.fromFuture(
                                                                    grpc
                                                                            .getMultimodalCacheStatusAsync(
                                                                                    worker.getIp(),
                                                                                    worker
                                                                                            .getGrpcPort(),
                                                                                    request,
                                                                                    2000)))
                                            .filter(response -> response.hasMultimodalCache())
                                            .doOnNext(
                                                    response ->
                                                            replace(
                                                                    worker,
                                                                    response.getMultimodalCache()))
                                            .onErrorResume(
                                                    error -> {
                                                        Logger.debug(
                                                                "ViT cache snapshot unavailable for {}: {}",
                                                                worker.getIpPort(),
                                                                error.toString());
                                                        return Mono.empty();
                                                    }),
                            4)
                    .then()
                    .block();
        } catch (Exception error) {
            Logger.warn("ViT cache sync failed: {}", error.toString());
        }
    }

    synchronized boolean replace(WorkerStatus worker, MultimodalCacheStatusPB response) {
        Map<String, WorkerStatus> live = workers.vitRoutingSnapshot();
        if (live.get(worker.getLogicalIpPort()) != worker
                || !isRoutable(worker, System.nanoTime() / 1000)
                || StringUtils.isBlank(response.getWorkerInstance())
                || !validKeys(response.getKeysList())
                || !validKeys(response.getGpuEmbeddingKeysList())
                || !validKeys(response.getCpuEmbeddingKeysList())) {
            return false;
        }
        Set<String> keys = Set.copyOf(response.getKeysList());
        Set<String> gpuKeys = Set.copyOf(response.getGpuEmbeddingKeysList());
        Set<String> cpuKeys = Set.copyOf(response.getCpuEmbeddingKeysList());
        Set<String> allKeys = new HashSet<>(keys);
        allKeys.addAll(gpuKeys);
        allKeys.addAll(cpuKeys);
        if (allKeys.size() > 100_000 || gpuKeys.stream().anyMatch(cpuKeys::contains)) {
            return false;
        }
        String address = worker.getLogicalIpPort();
        var updated = new HashMap<>(snapshots);
        updated.put(
                address,
                new Snapshot(
                        worker,
                        response.getWorkerInstance(),
                        keys,
                        gpuKeys,
                        cpuKeys,
                        System.currentTimeMillis()));
        snapshots = Map.copyOf(updated);
        return true;
    }

    private boolean validKeys(List<String> keys) {
        return keys == null
                || (keys.size() <= 100_000
                        && keys.stream()
                                .noneMatch(k -> k == null || k.isEmpty() || k.length() > 4096));
    }

    private void prune(Map<String, WorkerStatus> live, long now) {
        long nowUs = System.nanoTime() / 1000;
        var updated = new HashMap<>(snapshots);
        for (var entry : snapshots.entrySet()) {
            WorkerStatus worker = live.get(entry.getKey());
            if (worker != entry.getValue().worker() || !isRoutable(worker, nowUs)) {
                updated.remove(entry.getKey());
                attempts.remove(entry.getKey());
            } else if (!entry.getValue().isCurrent(worker, now)) {
                updated.remove(entry.getKey());
            }
        }
        if (updated.size() != snapshots.size()) {
            snapshots = Map.copyOf(updated);
        }
        attempts.keySet().removeIf(address -> !isRoutable(live.get(address), nowUs));
    }

    private static boolean isRoutable(WorkerStatus worker, long nowUs) {
        return worker != null
                && worker.isActiveGeneration()
                && worker.hasFreshSchedulingStatus(nowUs);
    }

    /**
     * Cache residency reported by one worker generation and valid for two refresh intervals.
     */
    private record Snapshot(
            WorkerStatus worker,
            String instance,
            Set<String> keys,
            Set<String> gpuEmbeddingKeys,
            Set<String> cpuEmbeddingKeys,
            long time) {
        boolean isCurrent(WorkerStatus currentWorker, long now) {
            return worker == currentWorker && now - time <= 2 * SYNC_INTERVAL_MS;
        }
    }

    public record Candidate(
            WorkerStatus worker,
            String instance,
            Set<String> hashKeys,
            int hashHits,
            int embeddingHits,
            int gpuHits) {}
}
