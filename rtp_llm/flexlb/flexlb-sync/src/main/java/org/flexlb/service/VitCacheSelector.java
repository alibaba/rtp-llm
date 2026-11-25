package org.flexlb.service;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.opentelemetry.api.trace.Span;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.RoutingConfig;
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
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/**
 * Selects ViT workers by cache affinity and inflight load, then revalidates the
 * selected endpoint identity against the full route's policy before admission.
 */
@Component
public class VitCacheSelector {
    private static final long MIN_PENDING_TTL_MS = 120_000;
    private static final long MAX_PENDING_TTL_MS = 600_000;
    private static final int MAX_PENDING = 100_000;

    private static final FlexMetricTags TOTAL_IMAGES = FlexMetricTags.of("kind", "total");
    private static final FlexMetricTags HIT_IMAGES = FlexMetricTags.of("kind", "hit");

    private final FlexMonitor monitor;
    private final VitCacheDirectory directory;
    private final WorkerDirectory workers;
    private final Cache<String, Placement> pending =
            Caffeine.newBuilder()
                    .maximumSize(MAX_PENDING)
                    .expireAfterWrite(MAX_PENDING_TTL_MS, TimeUnit.MILLISECONDS)
                    .build();
    private final ScheduledExecutorService maintenance =
            Executors.newSingleThreadScheduledExecutor(
                    r -> {
                        Thread thread = new Thread(r, "vit-placement-cleanup");
                        thread.setDaemon(true);
                        return thread;
                    });

    public VitCacheSelector(
            VitCacheDirectory directory, WorkerDirectory workers, FlexMonitor monitor) {
        this.monitor = monitor;
        this.directory = directory;
        this.workers = workers;
    }

    @PostConstruct
    public void start() {
        monitor.register(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, FlexMetricType.COUNTER);
        maintenance.scheduleWithFixedDelay(
                () -> {
                    try {
                        prunePending();
                    } catch (RuntimeException error) {
                        Logger.warn("ViT placement cleanup failed: {}", error.toString());
                    }
                },
                5,
                5,
                TimeUnit.SECONDS);
    }

    @PreDestroy
    public void stop() {
        maintenance.shutdownNow();
    }

    public void recordVitOnlyInputs(List<String> keys) {
        if (keys != null && !keys.isEmpty()) {
            monitor.report(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, TOTAL_IMAGES, keys.size());
            monitor.prepare(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, HIT_IMAGES);
        }
    }

    /** Holds the selected endpoint generation through the admission handoff. */
    public SelectedRole selectPinned(BalanceContext ctx, String group) {
        ServerStatus result =
                ctx.getRequest().getSelectedVit() == null
                        ? select(ctx, group)
                        : validate(ctx, group);
        if (!result.isSuccess()) {
            return null;
        }
        WorkerEndpoint.GenerationPin pin =
                workers.captureEndpoint(RoleType.VIT, result.getLogicalIpPort());
        if (pin == null) {
            return null;
        }
        if (pin.generationId() != result.getWorkerGeneration()) {
            pin.close();
            return null;
        }
        return SelectedRole.stateless(pin, result);
    }

    public ServerStatus select(BalanceContext ctx, String group) {
        return traceSelection(ctx, "select", span -> selectImpl(ctx, group, span));
    }

    /**
     * Revalidates the first selection against the full route's policy group and
     * current endpoint generation. Address equality alone cannot identify a worker
     * after replacement; generation zero is not a wildcard.
     */
    public ServerStatus validate(BalanceContext ctx, String group) {
        return traceSelection(ctx, "validate", span -> validateImpl(ctx, group, span));
    }

    void prunePending() {
        if (pending.estimatedSize() == 0) {
            return;
        }
        Map<String, Candidate> candidates = directory.candidates(null, List.of());
        long now = System.currentTimeMillis();
        pending.asMap()
                .entrySet()
                .removeIf(
                        entry ->
                                !entry.getValue()
                                        .matches(
                                                candidates.get(
                                                        entry.getValue()
                                                                .worker()
                                                                .getLogicalIpPort()),
                                                now));
        pending.cleanUp();
    }

    private ServerStatus selectImpl(BalanceContext ctx, String group, Span span) {
        RoutingConfig.VitConfig config =
                ctx.getConfig() == null
                        ? new RoutingConfig.VitConfig()
                        : ctx.getConfig().getRouter().getRoles().getVit();
        boolean affinity = config.isCacheAffinityEnabled();
        var keys = ctx.getRequest().getCacheAffinityKeys();
        if (keys == null) {
            keys = List.of();
        }
        if (keys.size() > 256
                || keys.stream().anyMatch(k -> k == null || k.isEmpty() || k.length() > 4096)) {
            return errorStatus(StrategyErrorType.INVALID_REQUEST);
        }
        // Charge the initial ViT route, once per media input (before key deduplication).
        int mediaCount = ctx.getRequest().isVitRouteOnly() ? keys.size() : 0;
        if (!affinity) {
            ServerStatus result;
            do {
                result = selectLeastInflight(ctx, group, config, mediaCount, span);
            } while (result == null);
            return result;
        }
        keys = new ArrayList<>(new LinkedHashSet<>(keys));
        Map<String, Candidate> candidates = directory.candidates(group, keys);
        // A cold key's placement is published atomically with its selection.
        synchronized (pending) {
            ServerStatus result;
            do {
                result = selectAffinity(ctx, group, config, keys, candidates, mediaCount, span);
            } while (result == null);
            return result;
        }
    }

    private static ServerStatus errorStatus(StrategyErrorType error) {
        ServerStatus status = new ServerStatus();
        status.setCode(error.getErrorCode());
        status.setMessage(error.getErrorMsg());
        return status;
    }

    private ServerStatus selectLeastInflight(
            BalanceContext ctx,
            String group,
            RoutingConfig.VitConfig config,
            int mediaCount,
            Span span) {
        WorkerStatus selected = null;
        WorkerStatus.VitLoad selectedSample = null;
        String address = null;
        long selectedRouted = 0;
        long bestLoad = Long.MAX_VALUE;
        int ties = 0;
        long nowNanos = System.nanoTime();
        for (var entry : workers.vitRoutingSnapshot()) {
            WorkerStatus worker = entry.endpoint().getStatus();
            if (group != null && !Objects.equals(group, worker.getGroup())) {
                continue;
            }
            WorkerStatus.VitLoad sample = worker.vitLoad();
            long routed = worker.vitRoutedInputs();
            long load = load(sample, routed, nowNanos, config);
            if (load < bestLoad) {
                bestLoad = load;
                ties = 0;
            }
            if (load == bestLoad && ThreadLocalRandom.current().nextInt(++ties) == 0) {
                selected = worker;
                selectedSample = sample;
                selectedRouted = routed;
                address = entry.address();
            }
        }
        ServerStatus result =
                checkedStatus(
                        selected, address, ctx.getRequestId(), group, selectedRouted, mediaCount);
        if (result != null && result.isSuccess()) {
            if (mediaCount > 0) {
                var snapshot = directory.snapshotFor(selected);
                rememberPlacement(
                        selected,
                        snapshot == null ? null : snapshot.instance(),
                        snapshot == null ? Set.of() : snapshot.keys(),
                        ctx.getRequest().getCacheAffinityKeys(),
                        ctx.getRequest().getGenerateTimeout());
            }
            recordLoad(span, selectedSample, nowNanos, config);
        }
        return result;
    }

    private static long load(
            WorkerStatus.VitLoad sample,
            long routed,
            long nowNanos,
            RoutingConfig.VitConfig config) {
        // Roll out FlexLB and the ViT workers in each routing group together. Older
        // workers omit mm_inflight_inputs and always rank behind workers with fresh samples.
        if (sample == null
                || nowNanos - sample.receivedNanos()
                        > TimeUnit.MILLISECONDS.toNanos(config.getLoadMaxAgeMs())) {
            return Long.MAX_VALUE;
        }
        long unreported = routed - sample.routedAtPollStart();
        return Math.min(sample.inflightInputs(), Long.MAX_VALUE - unreported) + unreported;
    }

    /** Null means another request claimed this observation; retry the load comparison. */
    private ServerStatus checkedStatus(
            WorkerStatus selected,
            String address,
            String requestId,
            String group,
            long observedRouted,
            int mediaCount) {
        if (selected == null) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        try (var pin = workers.captureEndpoint(RoleType.VIT, address)) {
            if (pin == null
                    || pin.endpoint().getStatus() != selected
                    || (group != null && !Objects.equals(group, selected.getGroup()))) {
                return errorStatus(StrategyErrorType.NO_VIT_WORKER);
            }
            if (mediaCount > 0 && !selected.tryAddVitRoutedInputs(observedRouted, mediaCount)) {
                return null;
            }
            return status(selected, requestId);
        }
    }

    private static ServerStatus status(WorkerStatus worker, String requestId) {
        ServerStatus status = new ServerStatus();
        status.setRole(RoleType.VIT);
        status.setWorkerGeneration(worker.getGenerationId());
        status.setServerIp(worker.getIp());
        status.setHttpPort(worker.getPort());
        status.setSelectedEngineIndex(worker.getEngineIndex(), worker.getMultiEngineNum());
        status.setGrpcPort(worker.getGrpcPort());
        status.setGroup(worker.getGroup());
        status.setRequestId(requestId);
        status.setSuccess(true);
        return status;
    }

    private void rememberPlacement(
            WorkerStatus worker,
            String instance,
            Set<String> confirmedHashKeys,
            List<String> keys,
            long timeout) {
        long ttl = Math.min(MAX_PENDING_TTL_MS, Math.max(MIN_PENDING_TTL_MS, timeout));
        Placement placement = new Placement(worker, instance, System.currentTimeMillis() + ttl);
        for (String key : keys) {
            if (!confirmedHashKeys.contains(key)) {
                pending.put(key, placement);
            }
        }
    }

    private static void recordLoad(
            Span span, WorkerStatus.VitLoad sample, long nowNanos, RoutingConfig.VitConfig config) {
        if (span == null) {
            return;
        }
        try {
            if (sample == null) {
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_LOAD_STATE, "missing");
                return;
            }
            long ageNanos = nowNanos - sample.receivedNanos();
            FlexlbTrace.setAttribute(
                    span, FlexlbTrace.VIT_MM_INFLIGHT_INPUTS, sample.inflightInputs());
            FlexlbTrace.setAttribute(
                    span,
                    FlexlbTrace.VIT_LOAD_SAMPLE_AGE_MS,
                    TimeUnit.NANOSECONDS.toMillis(Math.max(0, ageNanos)));
            FlexlbTrace.setAttribute(
                    span,
                    FlexlbTrace.VIT_LOAD_STATE,
                    ageNanos <= TimeUnit.MILLISECONDS.toNanos(config.getLoadMaxAgeMs())
                            ? "fresh"
                            : "stale");
        } catch (RuntimeException ignored) {
            // The unknown-load routing sentinel must never be exported as a media count.
        }
    }

    private ServerStatus selectAffinity(
            BalanceContext ctx,
            String group,
            RoutingConfig.VitConfig config,
            List<String> keys,
            Map<String, Candidate> candidates,
            int mediaCount,
            Span span) {
        long now = System.currentTimeMillis();
        long nowNanos = System.nanoTime();
        LoadedCandidate cacheLeader = null;
        LoadedCandidate leastLoaded = null;
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
            WorkerStatus.VitLoad sample = candidate.worker().vitLoad();
            long routed = candidate.worker().vitRoutedInputs();
            long load = load(sample, routed, nowNanos, config);
            var current =
                    new LoadedCandidate(
                            candidate,
                            sample,
                            routed,
                            score,
                            load,
                            ThreadLocalRandom.current().nextLong());
            if (cacheLeader == null
                    || LoadedCandidate.CACHE_FIRST.compare(current, cacheLeader) < 0) {
                cacheLeader = current;
            }
            if (leastLoaded == null
                    || LoadedCandidate.LOAD_FIRST.compare(current, leastLoaded) < 0) {
                leastLoaded = current;
            }
        }
        if (cacheLeader == null) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        LoadedCandidate selectedLoad = cacheLeader;
        if (cacheLeader.load() - leastLoaded.load() > config.getMaxExtraInflightInputs()) {
            selectedLoad = leastLoaded;
        }
        Candidate selected = selectedLoad.candidate();
        ServerStatus result =
                checkedStatus(
                        selected.worker(),
                        selected.worker().getLogicalIpPort(),
                        ctx.getRequestId(),
                        group,
                        selectedLoad.routedInputs(),
                        mediaCount);
        if (result == null || !result.isSuccess()) {
            return result;
        }
        if (mediaCount > 0) {
            // Count each original input against the winner before publishing this placement.
            int hits = 0;
            for (String key : ctx.getRequest().getCacheAffinityKeys()) {
                if (selected.hasCachedImage(key) || hasPendingPlacement(key, selected, now)) {
                    hits++;
                }
            }
            if (hits > 0) {
                monitor.report(MetricConstant.VIT_CACHE_AFFINITY_IMAGES, HIT_IMAGES, hits);
            }
        }
        rememberPlacement(
                selected.worker(),
                selected.instance(),
                selected.hashKeys(),
                keys,
                ctx.getRequest().getGenerateTimeout());
        recordLoad(span, selectedLoad.sample(), nowNanos, config);
        return result;
    }

    private boolean hasPendingPlacement(String key, Candidate candidate, long now) {
        Placement placement = pending.getIfPresent(key);
        return placement != null && placement.matches(candidate, now);
    }

    private ServerStatus validateImpl(BalanceContext ctx, String group, Span span) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null) {
            FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_VALIDATION_REASON, "missing_selection");
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        try (var pin = workers.captureEndpoint(RoleType.VIT, selected.getLogicalIpPort())) {
            WorkerStatus worker = pin == null ? null : pin.endpoint().getStatus();
            String reason = "valid";
            if (selected.getRole() != RoleType.VIT) {
                reason = "role_mismatch";
            } else if (worker == null) {
                reason = "endpoint_unavailable";
            } else if (group != null && !Objects.equals(group, worker.getGroup())) {
                reason = "policy_group_mismatch";
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_POLICY_GROUP, group);
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_CURRENT_GROUP, worker.getGroup());
            } else if (selected.getGrpcPort() != worker.getGrpcPort()) {
                reason = "grpc_port_mismatch";
                FlexlbTrace.setAttribute(
                        span,
                        FlexlbTrace.VIT_ADDRESS,
                        selected.getServerIp() + ":" + selected.getGrpcPort());
                FlexlbTrace.setAttribute(
                        span, FlexlbTrace.VIT_CURRENT_GRPC_PORT, (long) worker.getGrpcPort());
            } else if (!Objects.equals(selected.getGroup(), worker.getGroup())) {
                reason = "selected_group_mismatch";
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_SELECTED_GROUP, selected.getGroup());
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_CURRENT_GROUP, worker.getGroup());
            } else if (selected.getWorkerGeneration() != pin.generationId()) {
                reason = "generation_mismatch";
                FlexlbTrace.setAttribute(
                        span, FlexlbTrace.VIT_WORKER_GENERATION, selected.getWorkerGeneration());
                FlexlbTrace.setAttribute(
                        span, FlexlbTrace.VIT_CURRENT_GENERATION, pin.generationId());
            }
            if (!reason.equals("valid")) {
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_VALIDATION_REASON, reason);
                return errorStatus(StrategyErrorType.NO_VIT_WORKER);
            }
            // Validation does not reselect by load: report its current snapshot separately.
            if (span != null) {
                try {
                    RoutingConfig.VitConfig config =
                            ctx.getConfig() == null
                                    ? new RoutingConfig.VitConfig()
                                    : ctx.getConfig().getRouter().getRoles().getVit();
                    recordLoad(span, worker.vitLoad(), System.nanoTime(), config);
                } catch (RuntimeException ignored) {
                    // Telemetry must not affect endpoint validation.
                }
            }
            return status(worker, ctx.getRequestId());
        }
    }

    private ServerStatus traceSelection(
            BalanceContext ctx, String operation, Function<Span, ServerStatus> action) {
        if (!FlexlbTrace.isEnabled()) {
            return action.apply(null);
        }
        var parent = ctx == null ? null : ctx.getTraceContext();
        var span = FlexlbTrace.startInternal("rtp_llm.flexlb.vit." + operation, parent);
        var traceContext = span == null ? null : FlexlbTrace.withSpan(span, parent);
        try {
            if (ctx != null) {
                FlexlbTrace.setRequestAttributes(span, ctx.getRequestId());
                var request = ctx.getRequest();
                if (request != null) {
                    FlexlbTrace.setAttribute(
                            span, FlexlbTrace.VIT_ROUTE_ONLY, request.isVitRouteOnly());
                    FlexlbTrace.setAttribute(
                            span,
                            FlexlbTrace.VIT_KEY_COUNT,
                            request.getCacheAffinityKeys() == null
                                    ? 0L
                                    : (long) request.getCacheAffinityKeys().size());
                }
                FlexlbTrace.setAttribute(
                        span,
                        FlexlbTrace.VIT_CACHE_AFFINITY,
                        ctx.getConfig() != null
                                && ctx.getConfig()
                                        .getRouter()
                                        .getRoles()
                                        .getVit()
                                        .isCacheAffinityEnabled());
            }
        } catch (RuntimeException ignored) {
            // Observability must not change selector validation or admission.
        }
        try {
            ServerStatus result = action.apply(span);
            FlexlbTrace.setAttribute(span, FlexlbTrace.SCHEDULE_CODE, (long) result.getCode());
            if (result.isSuccess()) {
                FlexlbTrace.setAttribute(
                        span,
                        FlexlbTrace.VIT_ADDRESS,
                        result.getServerIp() + ":" + result.getGrpcPort());
                FlexlbTrace.setAttribute(
                        span, FlexlbTrace.VIT_WORKER_GENERATION, result.getWorkerGeneration());
            } else {
                FlexlbTrace.markBusinessError(
                        traceContext,
                        result.getCode(),
                        operation.equals("validate")
                                ? "VIT_ROUTE_STALE"
                                : "VIT_SELECTION_REJECTED");
            }
            return result;
        } catch (RuntimeException | Error error) {
            // Do not export exception messages: they can contain cache keys/URLs.
            FlexlbTrace.markBusinessError(traceContext, -1, "VIT_ROUTING_ERROR");
            throw error;
        } finally {
            FlexlbTrace.finish(span);
        }
    }

    private record Placement(WorkerStatus worker, String instance, long expiry) {
        boolean matches(Candidate candidate, long now) {
            return candidate != null
                    && worker == candidate.worker()
                    && expiry > now
                    && (instance == null || Objects.equals(instance, candidate.instance()));
        }
    }

    /**
     * Captures a candidate's cache score and effective load from one routing attempt. Its
     * comparators rank by cache hits or by load when the cache leader exceeds the load limit.
     */
    private record LoadedCandidate(
            Candidate candidate,
            WorkerStatus.VitLoad sample,
            long routedInputs,
            MatchScore score,
            long load,
            long tieBreaker) {
        private static final Comparator<LoadedCandidate> CACHE_FIRST =
                Comparator.comparing(LoadedCandidate::score, Comparator.reverseOrder())
                        .thenComparingLong(LoadedCandidate::load)
                        .thenComparingLong(LoadedCandidate::tieBreaker);
        private static final Comparator<LoadedCandidate> LOAD_FIRST =
                Comparator.comparingLong(LoadedCandidate::load).thenComparing(CACHE_FIRST);
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
}
