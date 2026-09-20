package org.flexlb.service;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService.CacheVersionPB;
import org.flexlb.engine.grpc.EngineRpcService.MultimodalCacheStatusPB;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

@Component
public class VitCacheDirectory {
    static final long SYNC_INTERVAL_MS = 600_000;
    private static final int MAX_PENDING = 100_000;
    private final EngineWorkerStatus workers;
    private final ConfigService config;
    private final ResourceMeasureFactory resources;
    private final EngineGrpcClient grpc;
    private final LBStatusConsistencyService consistency;
    private final Map<String, Snapshot> snapshots = new HashMap<>();
    private final Map<String, Set<String>> owners = new HashMap<>();
    private final LinkedHashMap<String, Placement> pending = new LinkedHashMap<>();
    private final Map<String, Long> attempts = new HashMap<>();
    private final ScheduledExecutorService sync = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread thread = new Thread(r, "vit-cache-sync");
        thread.setDaemon(true);
        return thread;
    });
    private boolean wasMaster;
    private long pendingPrunedAt;

    private record Snapshot(WorkerStatus worker, String instance, Set<String> keys,
                            Set<String> gpuEmbeddingKeys, Set<String> cpuEmbeddingKeys, long time) {}
    private record Placement(String worker, long expiry) {}

    private record MatchScore(int hashHits, int embeddingHits, int gpuHits, int pendingHits)
            implements Comparable<MatchScore> {
        @Override
        public int compareTo(MatchScore other) {
            int result = Integer.compare(hashHits, other.hashHits);
            if (result == 0) {
                result = Integer.compare(embeddingHits, other.embeddingHits);
            }
            if (result == 0) {
                result = Integer.compare(gpuHits, other.gpuHits);
            }
            return result == 0 ? Integer.compare(pendingHits, other.pendingHits) : result;
        }
    }

    public VitCacheDirectory(EngineWorkerStatus workers, ConfigService config,
                             ResourceMeasureFactory resources, EngineGrpcClient grpc,
                             LBStatusConsistencyService consistency) {
        this.workers = workers;
        this.config = config;
        this.resources = resources;
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
            Map<String, WorkerStatus> live = workers.selectModelWorkerStatus(RoleType.VIT, null);
            List<WorkerStatus> due = new ArrayList<>();
            long now = System.currentTimeMillis();
            synchronized (this) {
                prune(live, now);
                if (!wasMaster) {
                    attempts.clear();
                }
                for (WorkerStatus worker : live.values()) {
                    if (worker.isAlive() && now - attempts.getOrDefault(worker.getIpPort(), 0L) >= SYNC_INTERVAL_MS) {
                        attempts.put(worker.getIpPort(), now);
                        due.add(worker);
                    }
                }
                wasMaster = true;
            }
            CacheVersionPB request = CacheVersionPB.newBuilder().setNeedCacheKeys(true).build();
            Flux.fromIterable(due).flatMap(worker -> Mono.fromCallable(() -> grpc.getMultimodalCacheStatus(
                            worker.getIp(), CommonUtils.toGrpcPort(worker.getPort()), request, 2000))
                    .subscribeOn(Schedulers.boundedElastic())
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

    synchronized void replace(WorkerStatus worker, MultimodalCacheStatusPB response) {
        Map<String, WorkerStatus> live = workers.selectModelWorkerStatus(RoleType.VIT, null);
        if (live.get(worker.getIpPort()) != worker || !worker.isAlive()
                || StringUtils.isBlank(response.getWorkerInstance())
                || !validKeys(response.getKeysList())
                || !validKeys(response.getGpuEmbeddingKeysList()) || !validKeys(response.getCpuEmbeddingKeysList())) {
            return;
        }
        Set<String> keys = Set.copyOf(response.getKeysList());
        Set<String> gpuKeys = Set.copyOf(response.getGpuEmbeddingKeysList());
        Set<String> cpuKeys = Set.copyOf(response.getCpuEmbeddingKeysList());
        Set<String> allKeys = new HashSet<>(keys);
        allKeys.addAll(gpuKeys);
        allKeys.addAll(cpuKeys);
        if (allKeys.size() > 100_000 || gpuKeys.stream().anyMatch(cpuKeys::contains)) {
            return;
        }
        String address = worker.getIpPort();
        Snapshot previous = snapshots.get(address);
        if (previous != null && !previous.instance().equals(response.getWorkerInstance())) {
            pending.values().removeIf(p -> p.worker().equals(address));
        }
        removeSnapshot(address);
        snapshots.put(address, new Snapshot(worker, response.getWorkerInstance(), keys,
                gpuKeys, cpuKeys, System.currentTimeMillis()));
        for (String key : keys) {
            owners.computeIfAbsent(key, ignored -> new HashSet<>()).add(address);
        }
    }

    private boolean validKeys(List<String> keys) {
        return keys == null || (keys.size() <= 100_000
                && keys.stream().noneMatch(k -> k == null || k.isEmpty() || k.length() > 4096));
    }

    private void removeSnapshot(String address) {
        Snapshot old = snapshots.remove(address);
        if (old == null) {
            return;
        }
        for (String key : old.keys()) {
            Set<String> locations = owners.get(key);
            locations.remove(address);
            if (locations.isEmpty()) {
                owners.remove(key);
            }
        }
    }

    private void prune(Map<String, WorkerStatus> live, long now) {
        for (var entry : new ArrayList<>(snapshots.entrySet())) {
            WorkerStatus worker = live.get(entry.getKey());
            if (worker == null || !worker.isAlive() || worker != entry.getValue().worker()
                    || now - entry.getValue().time() > 2 * SYNC_INTERVAL_MS) {
                removeSnapshot(entry.getKey());
                pending.values().removeIf(p -> p.worker().equals(entry.getKey()));
                attempts.remove(entry.getKey());
            }
        }
        attempts.keySet().removeIf(address -> !live.containsKey(address) || !live.get(address).isAlive());
        if (now - pendingPrunedAt >= 5000) {
            pending.values().removeIf(p -> p.expiry() <= now || !live.containsKey(p.worker())
                    || !live.get(p.worker()).isAlive());
            pendingPrunedAt = now;
        }
    }

    private boolean available(BalanceContext ctx, WorkerStatus worker) {
        if (worker == null || !worker.isAlive()) {
            return false;
        }
        var cfg = ctx.getConfig() == null ? config.loadBalanceConfig() : ctx.getConfig();
        var measure = resources.getMeasure(cfg.getResourceMeasureIndicator(RoleType.VIT));
        return measure == null || measure.isResourceAvailable(worker);
    }

    public synchronized ServerStatus select(BalanceContext ctx, String group) {
        var keys = ctx.getRequest().getMediaKeys();
        if (keys == null || keys.isEmpty() || keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return ServerStatus.code(StrategyErrorType.INVALID_REQUEST);
        }
        keys = new ArrayList<>(new HashSet<>(keys));
        long now = System.currentTimeMillis();
        prune(workers.selectModelWorkerStatus(RoleType.VIT, null), now);
        List<WorkerStatus> best = new ArrayList<>();
        MatchScore bestScore = null;
        for (WorkerStatus worker : workers.selectModelWorkerStatus(RoleType.VIT, group).values()) {
            if (!available(ctx, worker)) {
                continue;
            }
            int hashHits = 0;
            int gpuHits = 0;
            int cpuHits = 0;
            int pendingHits = 0;
            Snapshot snapshot = snapshots.get(worker.getIpPort());
            for (String key : keys) {
                if (owners.getOrDefault(key, Set.of()).contains(worker.getIpPort())) {
                    hashHits++;
                }
                if (snapshot != null) {
                    if (snapshot.gpuEmbeddingKeys().contains(key)) {
                        gpuHits++;
                    } else if (snapshot.cpuEmbeddingKeys().contains(key)) {
                        cpuHits++;
                    }
                }
                Placement p = pending.get(key);
                if (p != null && p.expiry() > now && p.worker().equals(worker.getIpPort())) {
                    pendingHits++;
                }
            }
            MatchScore score = new MatchScore(hashHits, gpuHits + cpuHits, gpuHits, pendingHits);
            int comparison = bestScore == null ? 1 : score.compareTo(bestScore);
            if (comparison > 0) {
                best.clear();
                bestScore = score;
            }
            if (comparison >= 0) {
                best.add(worker);
            }
        }
        if (best.isEmpty()) {
            return ServerStatus.code(StrategyErrorType.NO_VIT_WORKER);
        }
        WorkerStatus selected = best.get(ThreadLocalRandom.current().nextInt(best.size()));
        long timeout = ctx.getRequest().getGenerateTimeout();
        long ttl = timeout <= 0 ? SYNC_INTERVAL_MS : Math.min(SYNC_INTERVAL_MS, Math.max(1000, timeout));
        for (String key : keys) {
            if (!owners.getOrDefault(key, Set.of()).contains(selected.getIpPort())) {
                pending.put(key, new Placement(selected.getIpPort(), now + ttl));
            }
        }
        while (pending.size() > MAX_PENDING) {
            pending.remove(pending.keySet().iterator().next());
        }
        return status(selected, ctx.getRequestId());
    }

    public synchronized ServerStatus validate(BalanceContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        WorkerStatus worker = workers.selectModelWorkerStatus(RoleType.VIT, group)
                .get(selected.getServerIp() + ":" + selected.getHttpPort());
        if (selected.getRole() != RoleType.VIT || !available(ctx, worker)
                || selected.getGrpcPort() != CommonUtils.toGrpcPort(worker.getPort())
                || !Objects.equals(selected.getGroup(), worker.getGroup())) {
            return ServerStatus.code(StrategyErrorType.NO_VIT_WORKER);
        }
        return status(worker, ctx.getRequestId());
    }

    private ServerStatus status(WorkerStatus worker, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.VIT);
        status.setServerIp(worker.getIp());
        status.setHttpPort(worker.getPort());
        status.setGrpcPort(CommonUtils.toGrpcPort(worker.getPort()));
        status.setGroup(worker.getGroup());
        status.setRequestId(requestId);
        status.setSuccess(true);
        Snapshot snapshot = snapshots.get(worker.getIpPort());
        status.setWorkerInstance(snapshot == null ? null : snapshot.instance());
        return status;
    }
}
