package org.flexlb.service;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerIdentity;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheDirectory.Candidate;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/** Least-inflight-media routing, optionally preceded by cache affinity. */
@Component
public class VitCacheSelector {
    private static final long MIN_PENDING_TTL_MS = 120_000;
    private static final long MAX_PENDING_TTL_MS = 600_000;
    private static final int MAX_PENDING = 100_000;
    private final VitCacheDirectory directory;
    private final WorkerDirectory workers;
    private final Cache<String, Placement> pending = Caffeine.newBuilder()
            .maximumSize(MAX_PENDING)
            .expireAfterWrite(MAX_PENDING_TTL_MS, TimeUnit.MILLISECONDS)
            .build();
    private final ScheduledExecutorService maintenance = Executors.newSingleThreadScheduledExecutor(r -> {
        Thread thread = new Thread(r, "vit-placement-cleanup");
        thread.setDaemon(true);
        return thread;
    });

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

    @PostConstruct
    public void start() {
        maintenance.scheduleWithFixedDelay(() -> {
            try {
                prunePending();
            } catch (RuntimeException error) {
                Logger.warn("ViT placement cleanup failed: {}", error.toString());
            }
        }, 5, 5, TimeUnit.SECONDS);
    }

    @PreDestroy
    public void stop() {
        maintenance.shutdownNow();
    }

    void prunePending() {
        if (pending.estimatedSize() == 0) {
            return;
        }
        Map<String, Candidate> candidates = directory.candidates(null, List.of());
        long now = System.currentTimeMillis();
        pending.asMap().entrySet().removeIf(entry ->
                !entry.getValue().matches(candidates.get(entry.getValue().worker().getIpPort()), now));
        pending.cleanUp();
    }

    public ServerStatus select(BalanceContext ctx, String group) {
        RoutingConfig.VitConfig config = ctx.getConfig() == null
                ? new RoutingConfig.VitConfig() : ctx.getConfig().getRouter().getRoles().getVit();
        boolean affinity = config.isCacheAffinityEnabled();
        var keys = ctx.getRequest().getCacheAffinityKeys();
        if (keys == null) { keys = List.of(); }
        if (keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return errorStatus(StrategyErrorType.INVALID_REQUEST);
        }
        if (!affinity) {
            return selectLeastInflight(ctx, group, config);
        }
        keys = new ArrayList<>(new LinkedHashSet<>(keys));
        Map<String, Candidate> candidates = directory.candidates(group, keys);
        // A cold key's placement is published atomically with its selection.
        synchronized (pending) {
            return selectAffinity(ctx, group, config, keys, candidates);
        }
    }

    private ServerStatus selectLeastInflight(
            BalanceContext ctx, String group, RoutingConfig.VitConfig config) {
        WorkerStatus selected = null;
        String address = null;
        long bestLoad = Long.MAX_VALUE;
        int ties = 0;
        long nowNanos = System.nanoTime();
        for (var entry : workers.vitRoutingSnapshot()) {
            WorkerStatus worker = entry.endpoint().getStatus();
            if (group != null && !Objects.equals(group, worker.getGroup())) {
                continue;
            }
            long load = load(worker, nowNanos, config);
            if (load < bestLoad) {
                bestLoad = load;
                ties = 0;
            }
            if (load == bestLoad && ThreadLocalRandom.current().nextInt(++ties) == 0) {
                selected = worker;
                address = entry.address();
            }
        }
        return checkedStatus(selected, address, ctx.getRequestId(), group);
    }

    private static long load(WorkerStatus worker, long nowNanos, RoutingConfig.VitConfig config) {
        WorkerStatus.VitLoad sample = worker.vitLoad();
        return sample != null && nowNanos - sample.receivedNanos()
                <= TimeUnit.MILLISECONDS.toNanos(config.getLoadMaxAgeMs())
                ? sample.inflightInputs() : Long.MAX_VALUE;
    }

    private ServerStatus selectAffinity(BalanceContext ctx, String group,
            RoutingConfig.VitConfig config, List<String> keys, Map<String, Candidate> candidates) {
        long now = System.currentTimeMillis();
        long nowNanos = System.nanoTime();

        List<Candidate> best = new ArrayList<>();
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
            MatchScore score = new MatchScore(
                    candidate.hashHits(), candidate.embeddingHits(), candidate.gpuHits(), pendingHits);
            long load = load(candidate.worker(), nowNanos, config);
            int comparison = bestScore == null ? 1 : score.compareTo(bestScore);
            if (comparison == 0) { comparison = Long.compare(bestLoad, load); }
            if (comparison > 0) {
                best.clear();
                bestScore = score;
                bestLoad = load;
            }
            if (comparison >= 0) {
                best.add(candidate);
            }
        }
        if (best.isEmpty()) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        Candidate selected = best.get(ThreadLocalRandom.current().nextInt(best.size()));
        ServerStatus result = checkedStatus(selected.worker(), selected.worker().getIpPort(), ctx.getRequestId(), group);
        if (!result.isSuccess()) {
            return result;
        }
        long timeout = ctx.getRequest().getGenerateTimeout();
        long ttl = timeout <= 0 ? MIN_PENDING_TTL_MS
                : Math.min(MAX_PENDING_TTL_MS, Math.max(MIN_PENDING_TTL_MS, timeout));
        for (String key : keys) {
            if (!selected.hashKeys().contains(key)) {
                pending.put(key, new Placement(selected.worker(), selected.instance(), now + ttl));
            }
        }
        return result;
    }

    private ServerStatus checkedStatus(WorkerStatus selected, String address, long requestId, String group) {
        if (selected == null) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        try (var pin = workers.captureEndpoint(RoleType.VIT, address)) {
            if (pin == null || pin.endpoint().getStatus() != selected
                    || (group != null && !Objects.equals(group, selected.getGroup()))) {
                return errorStatus(StrategyErrorType.NO_VIT_WORKER);
            }
            return status(selected, requestId);
        }
    }

    /**
     * Revalidates the first selection against the full route's policy group and
     * current endpoint generation. Address equality alone cannot identify a worker
     * after replacement; generation zero is not a wildcard.
     */
    public ServerStatus validate(BalanceContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null || selected.getServerIp() == null || selected.getHttpPort() <= 0) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        try (var pin = workers.captureEndpoint(RoleType.VIT,
                new WorkerIdentity(selected.getServerIp(), selected.getHttpPort(),
                        selected.getEngineIndex() == null ? 0 : selected.getEngineIndex()).getLogicalIpPort())) {
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

    /**
     * Holds the selected endpoint generation through the admission handoff.
     */
    public SelectedRole selectPinned(BalanceContext ctx, String group) {
        ServerStatus result = ctx.getRequest().getSelectedVit() == null ? select(ctx, group) : validate(ctx, group);
        if (!result.isSuccess()) {
            return null;
        }
        WorkerEndpoint.GenerationPin pin = workers.captureEndpoint(RoleType.VIT,
                result.getLogicalIpPort());
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

    private static ServerStatus status(WorkerStatus worker, String requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.VIT);
        status.setWorkerGeneration(worker.getGenerationId());
        status.setServerIp(worker.getIp());
        status.setHttpPort(worker.getPort());
        status.setGrpcPort(worker.getGrpcPort());
        status.setGroup(worker.getGroup());
        status.setSelectedEngineIndex(worker.getEngineIndex(), worker.getMultiEngineNum());
        status.setRequestId(requestId);
        status.setSuccess(true);
        return status;
    }
}
