package org.flexlb.service;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.consistency.LBStatusConsistencyService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.transport.GeneralHttpNettyService;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.net.URI;
import java.time.Duration;
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
    private final WorkerDirectory workers;
    private final GeneralHttpNettyService http;
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

    private record Snapshot(WorkerStatus worker, String instance, Set<String> keys, long time) {}
    private record Placement(String worker, long expiry) {}

    @Data
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class CacheKeys {
        @JsonProperty("worker_instance")
        private String workerInstance;
        @JsonProperty("feature_hash_version")
        private int featureHashVersion;
        private List<String> keys;
    }

    public VitCacheDirectory(WorkerDirectory workers, GeneralHttpNettyService http,
                             LBStatusConsistencyService consistency) {
        this.workers = workers;
        this.http = http;
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
            Map<String, WorkerStatus> live = workers.statusSnapshot(RoleType.VIT);
            Set<String> routable = Set.copyOf(
                    workers.endpointAddressSnapshot(RoleType.VIT));
            List<WorkerStatus> due = new ArrayList<>();
            long now = System.currentTimeMillis();
            synchronized (this) {
                prune(live, routable, now);
                if (!wasMaster) {
                    attempts.clear();
                }
                for (String address : routable) {
                    WorkerStatus worker = live.get(address);
                    if (worker != null
                            && now - attempts.getOrDefault(address, 0L)
                                    >= SYNC_INTERVAL_MS) {
                        attempts.put(worker.getIpPort(), now);
                        due.add(worker);
                    }
                }
                wasMaster = true;
            }
            Flux.fromIterable(due).flatMap(worker -> http.request(Map.of(),
                    URI.create("http://" + worker.getIpPort()), "/mm_cache/keys", CacheKeys.class)
                    .timeout(Duration.ofSeconds(2))
                    .doOnNext(keys -> replace(worker, keys))
                    .onErrorResume(error -> {
                        Logger.debug("ViT cache snapshot unavailable for {}: {}", worker.getIpPort(), error.toString());
                        return Mono.empty();
                    }), 4).then().block();
        } catch (Exception error) {
            Logger.warn("ViT cache sync failed: {}", error.toString());
        }
    }

    synchronized void replace(WorkerStatus worker, CacheKeys response) {
        if (!isCurrentRoutable(worker)
                || StringUtils.isBlank(response.getWorkerInstance()) || response.getFeatureHashVersion() != 1
                || response.getKeys() == null || response.getKeys().size() > 100_000
                || response.getKeys().stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return;
        }
        String address = worker.getIpPort();
        Snapshot previous = snapshots.get(address);
        if (previous != null && !previous.instance().equals(response.getWorkerInstance())) {
            pending.values().removeIf(p -> p.worker().equals(address));
        }
        removeSnapshot(address);
        Set<String> keys = Set.copyOf(response.getKeys());
        snapshots.put(address, new Snapshot(worker, response.getWorkerInstance(), keys, System.currentTimeMillis()));
        for (String key : keys) {
            owners.computeIfAbsent(key, ignored -> new HashSet<>()).add(address);
        }
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

    private void prune(
            Map<String, WorkerStatus> live,
            Set<String> routable,
            long now) {
        for (var entry : new ArrayList<>(snapshots.entrySet())) {
            WorkerStatus worker = live.get(entry.getKey());
            if (worker == null || worker != entry.getValue().worker()
                    || !routable.contains(entry.getKey())
                    || now - entry.getValue().time() > 2 * SYNC_INTERVAL_MS) {
                removeSnapshot(entry.getKey());
                pending.values().removeIf(p -> p.worker().equals(entry.getKey()));
                attempts.remove(entry.getKey());
            }
        }
        attempts.keySet().removeIf(address -> !routable.contains(address));
        if (now - pendingPrunedAt >= 5000) {
            pending.values().removeIf(p -> p.expiry() <= now
                    || !routable.contains(p.worker()));
            pendingPrunedAt = now;
        }
    }

    private boolean isCurrentRoutable(WorkerStatus worker) {
        if (worker == null
                || !workers.isCurrentStatus(
                        RoleType.VIT, worker.getIpPort(), worker)) {
            return false;
        }
        try (WorkerEndpoint.GenerationPin pin =
                     workers.captureEndpoint(RoleType.VIT, worker.getIpPort())) {
            return pin != null && pin.endpoint().getStatus() == worker;
        }
    }

    public synchronized PlacementResult<SelectedRole, RoleType> select(
            BalanceContext ctx, String group) {
        var keys = ctx.getRequest().getMediaKeys();
        if (keys == null || keys.isEmpty() || keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return PlacementResult.rejected(
                    Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        keys = new ArrayList<>(new HashSet<>(keys));
        long now = System.currentTimeMillis();
        Map<String, WorkerStatus> live = workers.statusSnapshot(RoleType.VIT);
        Set<String> routable = Set.copyOf(
                workers.endpointAddressSnapshot(RoleType.VIT));
        prune(live, routable, now);
        List<WorkerEndpoint.GenerationPin> best = new ArrayList<>();
        long bestScore = -1;
        for (String address : routable) {
            WorkerEndpoint.GenerationPin pin =
                    workers.captureEndpoint(RoleType.VIT, address);
            if (pin == null) {
                continue;
            }
            WorkerStatus worker = pin.endpoint().getStatus();
            WorkerStatus.TopologySnapshot topology = worker.topologySnapshot();
            if (live.get(address) != worker
                    || group != null && !group.equals(topology.group())) {
                pin.close();
                continue;
            }
            long score = 0;
            for (String key : keys) {
                if (owners.getOrDefault(key, Set.of()).contains(worker.getIpPort())) {
                    score += 257;
                }
                Placement p = pending.get(key);
                if (p != null && p.expiry() > now && p.worker().equals(worker.getIpPort())) {
                    score++;
                }
            }
            if (score > bestScore) {
                closePins(best);
                best.clear();
                bestScore = score;
            }
            if (score == bestScore) {
                best.add(pin);
            } else {
                pin.close();
            }
        }
        if (best.isEmpty()) {
            return PlacementResult.blocked(RoleType.VIT);
        }
        int selectedIndex = ThreadLocalRandom.current().nextInt(best.size());
        WorkerEndpoint.GenerationPin selectedPin = best.get(selectedIndex);
        WorkerStatus selected = selectedPin.endpoint().getStatus();
        for (int index = 0; index < best.size(); index++) {
            if (index != selectedIndex) {
                best.get(index).close();
            }
        }
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
        return PlacementResult.success(SelectedRole.stateless(
                selectedPin, status(selected, ctx.getRequestId())));
    }

    public synchronized SelectedRole validate(BalanceContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null || selected.getRole() != RoleType.VIT) {
            return null;
        }
        String address = selected.getServerIp() + ":" + selected.getHttpPort();
        WorkerEndpoint.GenerationPin pin =
                workers.captureEndpoint(RoleType.VIT, address);
        if (pin == null) {
            return null;
        }
        try {
            WorkerStatus worker = pin.endpoint().getStatus();
            WorkerStatus.TopologySnapshot topology = worker.topologySnapshot();
            if (!workers.isCurrentStatus(RoleType.VIT, address, worker)
                    || group != null && !group.equals(topology.group())
                    || selected.getGrpcPort() != topology.grpcPort()
                    || !Objects.equals(selected.getGroup(), topology.group())) {
                return null;
            }
            WorkerEndpoint.GenerationPin selectedPin = pin;
            pin = null;
            return SelectedRole.stateless(
                    selectedPin, status(worker, ctx.getRequestId()));
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }

    private ServerStatus status(WorkerStatus worker, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.VIT);
        status.setServerIp(worker.getIp());
        status.setHttpPort(worker.getPort());
        status.setGrpcPort(worker.getGrpcPort());
        status.setGroup(worker.getGroup());
        status.setDpRank(worker.committedEngineObservation().dpRank());
        status.setRequestId(requestId);
        status.setSuccess(true);
        Snapshot snapshot = snapshots.get(worker.getIpPort());
        status.setWorkerInstance(snapshot == null ? null : snapshot.instance());
        return status;
    }

    private static void closePins(
            List<WorkerEndpoint.GenerationPin> pins) {
        for (WorkerEndpoint.GenerationPin pin : pins) {
            pin.close();
        }
    }
}
