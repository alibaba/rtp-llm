package org.flexlb.service;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalCacheStatusPB;
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

@Component
public class VitCacheDirectory {
    // Cache residency changes after each image; ten-minute snapshots lose affinity.
    static final long SYNC_INTERVAL_MS = 30_000;
    private final WorkerDirectory workers;
    private final EngineGrpcClient grpc;
    private final LBStatusConsistencyService consistency;
    private volatile Map<String, Snapshot> snapshots = Map.of();
    private final Map<String, Long> attempts = new HashMap<>();
    private final ScheduledExecutorService sync = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread thread = new Thread(r, "vit-cache-sync");
        thread.setDaemon(true);
        return thread;
    });
    private boolean wasMaster;

    private record Snapshot(WorkerStatus worker, String instance, Set<String> keys,
                            Set<String> gpuEmbeddingKeys, Set<String> cpuEmbeddingKeys, long time) {}
    public record Candidate(WorkerStatus worker, String instance, Set<String> hashKeys,
                            int hashHits, int embeddingHits, int gpuHits) {}

    public VitCacheDirectory(WorkerDirectory workers, EngineGrpcClient grpc,
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

    void refresh() {
        try {
            if (consistency.isNeedConsistency() && !consistency.isMaster()) {
                wasMaster = false;
                return;
            }
            Map<String, WorkerStatus> live = liveWorkers();
            List<WorkerStatus> due = new ArrayList<>();
            long now = System.currentTimeMillis();
            synchronized (this) {
                prune(live, now);
                if (!wasMaster) {
                    attempts.clear();
                }
                for (WorkerStatus worker : live.values()) {
                    if (isRoutable(worker) && now - attempts.getOrDefault(worker.getIpPort(), 0L) >= SYNC_INTERVAL_MS) {
                        attempts.put(worker.getIpPort(), now);
                        due.add(worker);
                    }
                }
                wasMaster = true;
            }
            CacheVersionPB request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
            Flux.fromIterable(due).flatMap(worker -> Mono.defer(() -> Mono.fromFuture(grpc.getMultimodalCacheStatusAsync(
                            worker.getIp(), worker.getGrpcPort(), request, 2000)))
                    .filter(response -> response.hasMultimodalCache())
                    .doOnNext(response -> replace(worker, response.getMultimodalCache()))
                    .onErrorResume(error -> {
                        Logger.debug("ViT cache snapshot unavailable for {}: {}", worker.getIpPort(), error.toString());
                        return Mono.empty();
                    }), 4).then().block();
        } catch (Exception error) {
            Logger.warn("ViT cache sync failed: {}", error.toString());
        }
    }

    synchronized boolean replace(WorkerStatus worker, MultimodalCacheStatusPB response) {
        Map<String, WorkerStatus> live = liveWorkers();
        if (live.get(worker.getIpPort()) != worker || !isRoutable(worker)
                || StringUtils.isBlank(response.getWorkerInstance())
                || !validKeys(response.getKeysList())
                || !validKeys(response.getGpuEmbeddingKeysList()) || !validKeys(response.getCpuEmbeddingKeysList())) {
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
        String address = worker.getIpPort();
        Map<String, Snapshot> updated = new HashMap<>(snapshots);
        updated.put(address, new Snapshot(worker, response.getWorkerInstance(), keys,
                gpuKeys, cpuKeys, System.currentTimeMillis()));
        snapshots = Map.copyOf(updated);
        return true;
    }

    private boolean validKeys(List<String> keys) {
        return keys == null || (keys.size() <= 100_000
                && keys.stream().noneMatch(k -> k == null || k.isEmpty() || k.length() > 4096));
    }

    private void prune(Map<String, WorkerStatus> live, long now) {
        Map<String, Snapshot> updated = new HashMap<>(snapshots);
        for (var entry : snapshots.entrySet()) {
            WorkerStatus worker = live.get(entry.getKey());
            if (worker == null || !isRoutable(worker) || worker != entry.getValue().worker()) {
                updated.remove(entry.getKey());
                attempts.remove(entry.getKey());
            } else if (now - entry.getValue().time() > 2 * SYNC_INTERVAL_MS) {
                updated.remove(entry.getKey());
            }
        }
        snapshots = Map.copyOf(updated);
        attempts.keySet().removeIf(address -> !live.containsKey(address) || !isRoutable(live.get(address)));
    }

    private Map<String, WorkerStatus> liveWorkers() {
        Map<String, WorkerStatus> live = new HashMap<>();
        for (var entry : workers.vitRoutingSnapshot()) {
            live.put(entry.address(), entry.endpoint().getStatus());
        }
        return live;
    }

    private boolean isRoutable(WorkerStatus worker) {
        if (worker == null) { return false; }
        try (var pin = workers.captureEndpoint(RoleType.VIT, worker.getIpPort())) {
            return pin != null && pin.endpoint().getStatus() == worker;
        }
    }

    /** Cache matches share immutable key sets from one directory view. */
    public Map<String, Candidate> candidates(String group, List<String> keys) {
        var fleet = workers.vitRoutingSnapshot();
        Map<String, Snapshot> cacheView = snapshots;
        long now = System.currentTimeMillis();
        Map<String, Candidate> matches = new HashMap<>();
        for (var entry : fleet) {
            WorkerStatus worker = entry.endpoint().getStatus();
            if (group != null && !Objects.equals(group, worker.getGroup())) {
                continue;
            }
            Snapshot snapshot = cacheView.get(entry.address());
            if (snapshot != null && (snapshot.worker() != worker
                    || now - snapshot.time() > 2 * SYNC_INTERVAL_MS)) {
                snapshot = null;
            }
            int hashHits = 0;
            int gpuHits = 0;
            int cpuHits = 0;
            List<String> matchingKeys = group == null || Objects.equals(group, worker.getGroup()) ? keys : List.of();
            for (String key : matchingKeys) {
                if (snapshot != null && snapshot.keys().contains(key)) {
                    hashHits++;
                }
                if (snapshot != null) {
                    if (snapshot.gpuEmbeddingKeys().contains(key)) {
                        gpuHits++;
                    } else if (snapshot.cpuEmbeddingKeys().contains(key)) {
                        cpuHits++;
                    }
                }
            }
            matches.put(worker.getIpPort(), new Candidate(worker,
                    snapshot == null ? null : snapshot.instance(),
                    snapshot == null ? Set.of() : snapshot.keys(), hashHits, gpuHits + cpuHits, gpuHits));
        }
        return Map.copyOf(matches);
    }
}
