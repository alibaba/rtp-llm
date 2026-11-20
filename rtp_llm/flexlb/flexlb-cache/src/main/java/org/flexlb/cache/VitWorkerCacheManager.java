package org.flexlb.cache;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.rpc.protocol.EngineRpcService.MultimodalCacheStatusPB;
import org.flexlb.worker.status.WorkerStatus;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Maintains local ViT worker cache snapshots and pending image reservations for cache-aware
 * selection. Expired snapshots and replaced worker generations cannot provide cache hits.
 */
@Component
public class VitWorkerCacheManager {
    private static final long MIN_PENDING_TTL_MS = 120_000;
    private static final long MAX_PENDING_TTL_MS = 600_000;
    private static final int MAX_PENDING = 100_000;
    private static final long SNAPSHOT_TTL_MS = 60_000;

    private final LinkedHashMap<String, Placement> pending = new LinkedHashMap<>();

    private long pendingPrunedAt;

    private final EndpointRegistry workers;
    private final Map<String, VitWorkerCacheStatus> snapshots = new HashMap<>();
    private final Map<String, Set<String>> owners = new HashMap<>();

    public VitWorkerCacheManager(EndpointRegistry workers) {
        this.workers = workers;
    }

    /**
     * Cache matches share immutable key sets from one view of the worker snapshots.
     */
    public synchronized Map<String, Candidate> candidates(String group, List<String> keys) {
        Map<String, WorkerStatus> live = liveWorkers();
        prune(live, System.currentTimeMillis());
        Map<String, Candidate> matches = new HashMap<>();
        for (WorkerStatus worker : live.values()) {
            if (!isRoutable(worker)) {
                continue;
            }
            VitWorkerCacheStatus snapshot = snapshots.get(worker.getLogicalIpPort());
            int hashHits = 0;
            int gpuHits = 0;
            int cpuHits = 0;
            List<String> matchingKeys =
                    group == null || Objects.equals(group, worker.getGroup()) ? keys : List.of();
            for (String key : matchingKeys) {
                if (owners.getOrDefault(key, Set.of()).contains(worker.getLogicalIpPort())) {
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
            matches.put(
                    worker.getLogicalIpPort(),
                    new Candidate(
                            worker,
                            snapshot == null ? null : snapshot.instance(),
                            snapshot == null ? Set.of() : snapshot.keys(),
                            hashHits,
                            gpuHits + cpuHits,
                            gpuHits));
        }
        long now = System.currentTimeMillis();
        if (now - pendingPrunedAt >= 5000) {
            pending.values()
                    .removeIf(p -> !p.matches(matches.get(p.worker().getLogicalIpPort()), now));
            pendingPrunedAt = now;
        }
        return Map.copyOf(matches);
    }

    public synchronized int pendingHits(Candidate candidate, List<String> keys, long now) {
        int hits = 0;
        for (String key : keys) {
            Placement placement = pending.get(key);
            if (placement != null && placement.matches(candidate, now)) hits++;
        }
        return hits;
    }

    public synchronized void reserve(
            List<String> keys, Candidate selected, long timeout, long now) {
        long ttl =
                timeout <= 0
                        ? MIN_PENDING_TTL_MS
                        : Math.clamp(timeout, MIN_PENDING_TTL_MS, MAX_PENDING_TTL_MS);
        for (String key : keys) {
            if (!selected.hashKeys().contains(key)) {
                pending.put(key, new Placement(selected.worker(), selected.instance(), now + ttl));
            }
        }
        while (pending.size() > MAX_PENDING) pending.pollFirstEntry();
    }

    synchronized boolean replace(WorkerStatus worker, MultimodalCacheStatusPB response) {
        Map<String, WorkerStatus> live = liveWorkers();
        if (live.get(worker.getLogicalIpPort()) != worker
                || !isRoutable(worker)
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
        removeSnapshot(address);
        snapshots.put(
                address,
                new VitWorkerCacheStatus(
                        worker,
                        response.getWorkerInstance(),
                        keys,
                        gpuKeys,
                        cpuKeys,
                        System.currentTimeMillis()));
        for (String key : keys) {
            owners.computeIfAbsent(key, ignored -> new HashSet<>()).add(address);
        }
        return true;
    }

    synchronized void prune(Map<String, WorkerStatus> live, long now) {
        for (var entry : new ArrayList<>(snapshots.entrySet())) {
            WorkerStatus worker = live.get(entry.getKey());
            if (worker == null || !isRoutable(worker) || worker != entry.getValue().worker()) {
                removeSnapshot(entry.getKey());
            } else if (now - entry.getValue().time() > SNAPSHOT_TTL_MS) {
                removeSnapshot(entry.getKey());
            }
        }
    }

    private boolean validKeys(List<String> keys) {
        return keys == null
                || (keys.size() <= 100_000
                        && keys.stream()
                                .noneMatch(k -> k == null || k.isEmpty() || k.length() > 4096));
    }

    private boolean isRoutable(WorkerStatus worker) {
        if (worker == null) {
            return false;
        }
        {
            var selectedEndpoint = workers.getEndpoint(RoleType.VIT, worker.getLogicalIpPort());
            return selectedEndpoint != null
                    && selectedEndpoint.acceptsNewRequests()
                    && selectedEndpoint.getStatus() == worker;
        }
    }

    private void removeSnapshot(String address) {
        VitWorkerCacheStatus old = snapshots.remove(address);
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

    private Map<String, WorkerStatus> liveWorkers() {
        Map<String, WorkerStatus> live = new HashMap<>();
        for (var endpoint : workers.endpoints(RoleType.VIT, null)) {
            if (!endpoint.acceptsNewRequests()) continue;
            String address = endpoint.ipPort();
            if (workers.getEndpoint(RoleType.VIT, address) == endpoint) {
                live.put(address, endpoint.getStatus());
            }
        }
        return live;
    }

    /**
     * Matches a cached ViT placement only while its worker identity and expiry remain valid.
     */
    private record Placement(WorkerStatus worker, String instance, long expiry) {
        boolean matches(Candidate candidate, long now) {
            return candidate != null
                    && worker == candidate.worker()
                    && expiry > now
                    && (instance == null || Objects.equals(instance, candidate.instance()));
        }
    }

    /**
     * Records one worker generation's image and embedding residency at the snapshot time.
     */
    private record VitWorkerCacheStatus(
            WorkerStatus worker,
            String instance,
            Set<String> keys,
            Set<String> gpuEmbeddingKeys,
            Set<String> cpuEmbeddingKeys,
            long time) {}

    /**
     * Exposes the Encoder availability and load inputs needed to select a worker.
     */
    public record Candidate(
            WorkerStatus worker,
            String instance,
            Set<String> hashKeys,
            int hashHits,
            int embeddingHits,
            int gpuHits) {}
}
