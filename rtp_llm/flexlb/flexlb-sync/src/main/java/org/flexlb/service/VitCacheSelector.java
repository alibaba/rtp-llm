package org.flexlb.service;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheDirectory.Candidate;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

/** Selects cache-affine ViT endpoints and owns temporary cold-key placements. */
@Component
public class VitCacheSelector {
    private static final long MIN_PENDING_TTL_MS = 120_000;
    private static final long MAX_PENDING_TTL_MS = 600_000;
    private static final int MAX_PENDING = 100_000;
    private final VitCacheDirectory directory;
    private final WorkerDirectory workers;
    private final LinkedHashMap<String, Placement> pending = new LinkedHashMap<>();
    private long pendingPrunedAt;

    private record Placement(WorkerStatus worker, String instance, long expiry) {
        boolean matches(Candidate candidate, long now) {
            return candidate != null && worker == candidate.worker() && expiry > now
                    && (instance == null || Objects.equals(instance, candidate.instance()));
        }
    }

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

    public VitCacheSelector(VitCacheDirectory directory, WorkerDirectory workers) {
        this.directory = directory;
        this.workers = workers;
    }

    public synchronized ServerStatus select(BalanceContext ctx, String group) {
        var keys = ctx.getRequest().getCacheAffinityKeys();
        if (keys == null || keys.isEmpty() || keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return errorStatus(StrategyErrorType.INVALID_REQUEST);
        }
        keys = new ArrayList<>(new LinkedHashSet<>(keys));
        Map<String, Candidate> candidates = directory.candidates(group, keys);
        long now = System.currentTimeMillis();
        if (now - pendingPrunedAt >= 5000) {
            pending.values().removeIf(p -> !p.matches(candidates.get(p.worker().getIpPort()), now));
            pendingPrunedAt = now;
        }
        List<Candidate> best = new ArrayList<>();
        MatchScore bestScore = null;
        for (Candidate candidate : candidates.values()) {
            if (group != null && !Objects.equals(group, candidate.worker().getGroup())) {
                continue;
            }
            int pendingHits = 0;
            for (String key : keys) {
                Placement placement = pending.get(key);
                if (placement != null && placement.matches(candidate, now)) {
                    pendingHits++;
                }
            }
            MatchScore score = new MatchScore(candidate.hashHits(), candidate.embeddingHits(),
                    candidate.gpuHits(), pendingHits);
            int comparison = bestScore == null ? 1 : score.compareTo(bestScore);
            if (comparison > 0) {
                best.clear();
                bestScore = score;
            }
            if (comparison >= 0) {
                best.add(candidate);
            }
        }
        if (best.isEmpty()) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        Candidate selected = best.get(ThreadLocalRandom.current().nextInt(best.size()));
        long timeout = ctx.getRequest().getGenerateTimeout();
        long ttl = timeout <= 0 ? MIN_PENDING_TTL_MS
                : Math.min(MAX_PENDING_TTL_MS, Math.max(MIN_PENDING_TTL_MS, timeout));
        for (String key : keys) {
            if (!selected.hashKeys().contains(key)) {
                pending.put(key, new Placement(selected.worker(), selected.instance(), now + ttl));
            }
        }
        while (pending.size() > MAX_PENDING) {
            pending.remove(pending.keySet().iterator().next());
        }
        return status(selected.worker(), ctx.getRequestId());
    }

    /**
     * Revalidates the first selection against the full route's policy group and
     * current endpoint generation. Address equality alone cannot identify a worker
     * after replacement; generation zero is not a wildcard.
     */
    public synchronized ServerStatus validate(BalanceContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        try (var pin = workers.captureEndpoint(RoleType.VIT,
                selected.getServerIp() + ":" + selected.getHttpPort())) {
            WorkerStatus worker = pin == null ? null : pin.endpoint().getStatus();
            if (selected.getRole() != RoleType.VIT || worker == null
                    || (group != null && !Objects.equals(group, worker.getGroup()))
                    || selected.getGrpcPort() != worker.getGrpcPort()
                    || !Objects.equals(selected.getGroup(), worker.getGroup())
                    || selected.getWorkerGeneration() != pin.generationId()) {
                return errorStatus(StrategyErrorType.NO_VIT_WORKER);
            }
            return status(worker, ctx.getRequestId());
        }
    }

    /** Holds the selected endpoint generation through the admission handoff. */
    public synchronized SelectedRole selectPinned(BalanceContext ctx, String group) {
        ServerStatus result = ctx.getRequest().getSelectedVit() == null ? select(ctx, group) : validate(ctx, group);
        if (!result.isSuccess()) {
            return null;
        }
        WorkerEndpoint.GenerationPin pin = workers.captureEndpoint(RoleType.VIT,
                result.getServerIp() + ":" + result.getHttpPort());
        if (pin == null) {
            return null;
        }
        if (pin.generationId() != result.getWorkerGeneration()) {
            pin.close();
            return null;
        }
        return SelectedRole.stateless(pin, result);
    }

    private static ServerStatus errorStatus(StrategyErrorType error) {
        ServerStatus status = new ServerStatus();
        status.setCode(error.getErrorCode());
        status.setMessage(error.getErrorMsg());
        return status;
    }

    private static ServerStatus status(WorkerStatus worker, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.VIT);
        status.setWorkerGeneration(worker.getGenerationId());
        status.setServerIp(worker.getIp());
        status.setHttpPort(worker.getPort());
        status.setGrpcPort(worker.getGrpcPort());
        status.setGroup(worker.getGroup());
        status.setRequestId(requestId);
        status.setSuccess(true);
        return status;
    }
}
