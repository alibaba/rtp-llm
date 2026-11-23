package org.flexlb.service;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheDirectory.Candidate;
import org.flexlb.sync.status.WorkerDirectory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;

/**
 * Scores ViT snapshots, pins the winner and shares temporary cold-key placements.
 * Background maintenance prunes placements as workers and cache instances change.
 */
@Component
public class VitCacheSelector {
    private static final long MIN_PENDING_TTL_MS = 120_000;
    private static final long MAX_PENDING_TTL_MS = 600_000;
    private static final int MAX_PENDING = 100_000;
    private static final long MAX_LOAD_AGE_NANOS = TimeUnit.SECONDS.toNanos(1);

    private final VitCacheDirectory directory;
    private final WorkerDirectory workers;
    private final Cache<String, Placement> pending =
            Caffeine.newBuilder()
                    .maximumSize(MAX_PENDING)
                    .expireAfterWrite(MAX_PENDING_TTL_MS, TimeUnit.MILLISECONDS)
                    .build();

    public VitCacheSelector(VitCacheDirectory directory, WorkerDirectory workers) {
        this.directory = directory;
        this.workers = workers;
    }

    /** Holds the selected endpoint generation through the admission handoff. */
    public SelectedRole selectPinned(BalanceContext ctx, String group) {
        if (ctx.getRequest().getSelectedVit() != null) {
            return pinSelectedWorker(ctx, group);
        }
        var keys = ctx.getRequest().getCacheAffinityKeys();
        return validKeys(ctx, keys) ? selectNewWorker(ctx, group, keys) : null;
    }

    public ServerStatus select(BalanceContext ctx, String group) {
        var keys = ctx.getRequest().getCacheAffinityKeys();
        if (!validKeys(ctx, keys)) {
            return errorStatus(StrategyErrorType.INVALID_REQUEST);
        }
        try (var selected = selectNewWorker(ctx, group, keys)) {
            return selected == null
                    ? errorStatus(StrategyErrorType.NO_VIT_WORKER)
                    : selected.serverStatus();
        }
    }

    /**
     * Revalidate the earlier route against its exact endpoint generation and the full route's group.
     */
    public ServerStatus validate(BalanceContext ctx, String group) {
        try (var selected = pinSelectedWorker(ctx, group)) {
            return selected == null
                    ? errorStatus(StrategyErrorType.NO_VIT_WORKER)
                    : selected.serverStatus();
        }
    }

    @Scheduled(fixedDelay = 5000)
    void prunePending() {
        if (pending.estimatedSize() == 0) {
            return;
        }
        var candidates = directory.candidates(null, List.of());
        long now = System.currentTimeMillis();
        pending.asMap()
                .forEach(
                        (key, placement) -> {
                            if (!placement.matches(
                                    candidates.get(placement.worker().routingAddress()), now)) {
                                pending.asMap().remove(key, placement);
                            }
                        });
        pending.cleanUp();
    }

    private static boolean validKeys(BalanceContext ctx, List<String> keys) {
        return keys != null
                && (!keys.isEmpty() || ctx.getRequest().isVitRouteOnly())
                && keys.size() <= 256
                && keys.stream().noneMatch(k -> k == null || k.isEmpty() || k.length() > 4096);
    }

    private SelectedRole selectNewWorker(
            BalanceContext ctx, String group, List<String> requestKeys) {
        int mediaCount = ctx.getRequest().isVitRouteOnly() ? requestKeys.size() : 0;
        var keys = List.copyOf(new LinkedHashSet<>(requestKeys));
        var candidates = directory.candidates(group, keys);
        if (keys.isEmpty()) {
            return rankAndPin(ctx, group, keys, candidates, mediaCount);
        }
        // Rank and record cold-key placements together so concurrent requests share a placement.
        synchronized (pending) {
            return rankAndPin(ctx, group, keys, candidates, mediaCount);
        }
    }

    private SelectedRole rankAndPin(
            BalanceContext ctx,
            String group,
            List<String> keys,
            Map<String, Candidate> candidates,
            int mediaCount) {
        long now = System.currentTimeMillis();
        long nowNanos = System.nanoTime();
        Candidate selected = null;
        int ties = 0;
        MatchScore bestScore = null;
        long bestLoad = Long.MAX_VALUE;
        for (Candidate candidate : candidates.values()) {
            if (group != null && !Objects.equals(group, candidate.worker().getGroup())) {
                continue;
            }
            int pendingHits = 0;
            for (String key : keys) {
                Placement placement = pending.getIfPresent(key);
                if (placement != null && placement.matches(candidate, now)) {
                    pendingHits++;
                }
            }
            MatchScore score =
                    new MatchScore(
                            candidate.hashHits(),
                            candidate.embeddingHits(),
                            candidate.gpuHits(),
                            pendingHits);
            int comparison = bestScore == null ? 1 : score.compareTo(bestScore);
            long load = candidate.worker().vitLoadEstimate(nowNanos, MAX_LOAD_AGE_NANOS);
            if (comparison == 0) {
                comparison = Long.compare(bestLoad, load);
            }
            if (comparison > 0) {
                ties = 0;
                bestScore = score;
                bestLoad = load;
            }
            if (comparison >= 0 && ThreadLocalRandom.current().nextInt(++ties) == 0) {
                selected = candidate;
            }
        }
        if (selected == null) {
            return null;
        }
        var worker = selected.worker();
        var pin = workers.captureEndpoint(RoleType.VIT, worker.routingAddress());
        if (pin == null) {
            return null;
        }
        if (pin.endpoint().getStatus() != worker
                || (group != null && !Objects.equals(group, worker.getGroup()))) {
            pin.close();
            return null;
        }
        try {
            var result = SelectedRole.stateless(pin, status(worker, ctx.getRequestId()));
            long timeout = ctx.getRequest().getGenerateTimeout();
            long ttl =
                    timeout <= 0
                            ? MIN_PENDING_TTL_MS
                            : Math.min(MAX_PENDING_TTL_MS, Math.max(MIN_PENDING_TTL_MS, timeout));
            for (String key : keys) {
                if (!selected.hashKeys().contains(key)) {
                    pending.put(key, new Placement(worker, selected.instance(), now + ttl));
                }
            }
            if (mediaCount > 0) {
                worker.recordVitRoute(mediaCount);
            }
            return result;
        } catch (RuntimeException | Error failure) {
            pin.close();
            throw failure;
        }
    }

    private static ServerStatus status(WorkerStatus worker, String requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.VIT);
        status.setWorkerGeneration(worker.getGenerationId());
        status.setServerIp(worker.getIp());
        status.setHttpPort(worker.getPort());
        status.setGrpcPort(worker.getGrpcPort());
        status.setSelectedEngineIndex(worker.getEngineIndex(), worker.getMultiEngineNum());
        status.setGroup(worker.getGroup());
        status.setRequestId(requestId);
        status.setSuccess(true);
        return status;
    }

    private SelectedRole pinSelectedWorker(BalanceContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null
                || selected.getRole() != RoleType.VIT
                || selected.getServerIp() == null
                || selected.getHttpPort() <= 0) {
            return null;
        }
        var pin = workers.captureEndpoint(RoleType.VIT, selected.getPhysicalIpPort());
        if (pin == null) {
            return null;
        }
        WorkerStatus worker = pin.endpoint().getStatus();
        if (worker == null
                || Objects.requireNonNullElse(selected.getEngineIndex(), 0) != worker.getEngineIndex()
                || (group != null && !Objects.equals(group, worker.getGroup()))
                || !Objects.equals(selected.getGrpcPort(), worker.getGrpcPort())
                || !Objects.equals(selected.getGroup(), worker.getGroup())
                || selected.getWorkerGeneration() != pin.generationId()) {
            pin.close();
            return null;
        }
        return SelectedRole.stateless(pin, status(worker, ctx.getRequestId()));
    }

    private static ServerStatus errorStatus(StrategyErrorType error) {
        ServerStatus status = new ServerStatus();
        status.setCode(error.getErrorCode());
        status.setMessage(error.getErrorMsg());
        return status;
    }

    /**
     * Checks whether a retained multimodal placement still matches the candidate generation and
     * expiry.
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
     * Orders multimodal candidates by the configured cache coverage and residency preference.
     */
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
}
