package org.flexlb.service;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.VitCacheDirectory.Candidate;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.telemetry.FlexlbTrace;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
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

    private static final FlexMetricTags TOTAL_IMAGES = FlexMetricTags.of("kind", "total");
    private static final FlexMetricTags HIT_IMAGES = FlexMetricTags.of("kind", "hit");

    private final VitCacheDirectory directory;
    private final WorkerDirectory workers;
    private final FlexMonitor monitor;
    private final Cache<String, Placement> pending =
            Caffeine.newBuilder()
                    .maximumSize(MAX_PENDING)
                    .expireAfterWrite(MAX_PENDING_TTL_MS, TimeUnit.MILLISECONDS)
                    .build();

    public VitCacheSelector(
            VitCacheDirectory directory, WorkerDirectory workers, FlexMonitor monitor) {
        this.directory = directory;
        this.workers = workers;
        this.monitor = monitor;
    }

    /** Count the original image inputs at the ViT-only route entry, including duplicate keys. */
    public void recordVitOnlyInputs(List<String> keys) {
        if (keys != null && !keys.isEmpty()) {
            monitor.report(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, TOTAL_IMAGES, keys.size());
            monitor.prepare(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, HIT_IMAGES);
        }
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

    @PostConstruct
    void registerMetrics() {
        monitor.register(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, FlexMetricType.COUNTER);
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
        var config = ctx.getConfig();
        if (requestKeys.isEmpty()
                || (config != null
                        && !config.getRouter().getRoles().getVit().isCacheAffinityEnabled())) {
            return selectByLoad(ctx, group, mediaCount);
        }
        var keys = List.copyOf(new LinkedHashSet<>(requestKeys));
        var candidates = directory.candidates(group, keys);
        // Rank and record cold-key placements together so concurrent requests share a placement.
        synchronized (pending) {
            return rankAndPin(ctx, group, keys, candidates, mediaCount);
        }
    }

    private SelectedRole selectByLoad(BalanceContext ctx, String group, int mediaCount) {
        while (true) {
            long nowNanos = System.nanoTime();
            long nowUs = TimeUnit.NANOSECONDS.toMicros(nowNanos);
            WorkerStatus selected = null;
            long selectedRoutedInputs = 0;
            long bestLoad = Long.MAX_VALUE;
            int ties = 0;
            for (WorkerStatus worker : workers.vitRoutingSnapshot().values()) {
                if (!VitCacheDirectory.isRoutableForGroup(worker, group, nowUs)) {
                    continue;
                }
                long routedInputs = worker.vitRoutedInputs();
                long load = worker.vitLoadEstimate(nowNanos, MAX_LOAD_AGE_NANOS, routedInputs);
                if (load < bestLoad) {
                    bestLoad = load;
                    ties = 0;
                }
                if (load == bestLoad && ThreadLocalRandom.current().nextInt(++ties) == 0) {
                    selected = worker;
                    selectedRoutedInputs = routedInputs;
                }
            }
            if (selected == null) {
                return null;
            }
            var result = pinRouteWorker(ctx, group, selected);
            if (result == null
                    || mediaCount == 0
                    || selected.tryRecordVitRoute(selectedRoutedInputs, mediaCount)) {
                return result;
            }
            // Another route changed the selected worker's load. Release its pin and rank again.
            result.close();
        }
    }

    private SelectedRole pinRouteWorker(BalanceContext ctx, String group, WorkerStatus worker) {
        var pin = workers.captureEndpoint(RoleType.VIT, worker.routingAddress());
        if (pin == null) {
            return null;
        }
        try {
            if (pin.endpoint().getStatus() == worker
                    && (group == null || Objects.equals(group, worker.getGroup()))) {
                return SelectedRole.stateless(pin, status(worker, ctx.getRequestId()));
            }
        } catch (RuntimeException | Error failure) {
            pin.close();
            throw failure;
        }
        pin.close();
        return null;
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
                if (hasPendingPlacement(key, candidate, now)) {
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
        var result = pinRouteWorker(ctx, group, worker);
        if (result == null) {
            return null;
        }
        try {
            if (mediaCount > 0) {
                reportTheoreticalHits(ctx.getRequest().getCacheAffinityKeys(), selected, now);
            }
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
            result.close();
            throw failure;
        }
    }

    private boolean hasPendingPlacement(String key, Candidate candidate, long now) {
        Placement placement = pending.getIfPresent(key);
        return placement != null && placement.matches(candidate, now);
    }

    private void reportTheoreticalHits(List<String> keys, Candidate selected, long now) {
        // Use the original input list and count before publishing this route's placements.
        int hits = 0;
        for (String key : keys) {
            if (selected.hasCachedImage(key) || hasPendingPlacement(key, selected, now)) {
                hits++;
            }
        }
        if (hits > 0) {
            monitor.report(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, HIT_IMAGES, hits);
        }
    }

    private SelectedRole pinSelectedWorker(BalanceContext ctx, String group) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null) {
            return rejectSelectedWorker(ctx, "missing_selection");
        }
        if (selected.getRole() != RoleType.VIT) {
            return rejectSelectedWorker(ctx, "role_mismatch");
        }
        if (selected.getServerIp() == null || selected.getHttpPort() <= 0) {
            return rejectSelectedWorker(ctx, "missing_address");
        }
        var pin = workers.captureEndpoint(RoleType.VIT, selected.getPhysicalIpPort());
        if (pin == null) {
            return rejectSelectedWorker(ctx, "endpoint_unavailable");
        }
        WorkerStatus worker = pin.endpoint().getStatus();
        String rejection = selectedWorkerRejection(selected, worker, group, pin.generationId());
        if (rejection != null) {
            pin.close();
            return rejectSelectedWorker(ctx, rejection);
        }
        return SelectedRole.stateless(pin, status(worker, ctx.getRequestId()));
    }

    private static SelectedRole rejectSelectedWorker(BalanceContext ctx, String reason) {
        FlexlbTrace.setScheduleAttribute(
                ctx.getTraceContext(), FlexlbTrace.VIT_VALIDATION_REASON, reason);
        return null;
    }

    private static String selectedWorkerRejection(
            ServerStatus selected, WorkerStatus worker, String group, long generation) {
        if (worker == null) {
            return "endpoint_unavailable";
        }
        if (Objects.requireNonNullElse(selected.getEngineIndex(), 0) != worker.getEngineIndex()) {
            return "engine_index_mismatch";
        }
        if (group != null && !Objects.equals(group, worker.getGroup())) {
            return "policy_group_mismatch";
        }
        if (!Objects.equals(selected.getGrpcPort(), worker.getGrpcPort())) {
            return "grpc_port_mismatch";
        }
        if (!Objects.equals(selected.getGroup(), worker.getGroup())) {
            return "selected_group_mismatch";
        }
        return selected.getWorkerGeneration() == generation ? null : "generation_mismatch";
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
