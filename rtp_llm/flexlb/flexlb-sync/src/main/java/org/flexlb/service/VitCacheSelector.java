package org.flexlb.service;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import io.opentelemetry.api.trace.Span;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheDirectory.Candidate;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.telemetry.FlexlbTrace;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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

    /** Keep the exact load sample used for this candidate's routing decision. */
    private record LoadedCandidate(Candidate candidate, WorkerStatus.VitLoad sample) {}

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
        return traceSelection(ctx, "select", span -> selectImpl(ctx, group, span));
    }

    private ServerStatus selectImpl(BalanceContext ctx, String group, Span span) {
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
            return selectLeastInflight(ctx, group, config, span);
        }
        keys = new ArrayList<>(new LinkedHashSet<>(keys));
        Map<String, Candidate> candidates = directory.candidates(group, keys);
        // A cold key's placement is published atomically with its selection.
        synchronized (pending) {
            return selectAffinity(ctx, group, config, keys, candidates, span);
        }
    }

    private ServerStatus selectLeastInflight(
            BalanceContext ctx, String group, RoutingConfig.VitConfig config, Span span) {
        WorkerStatus selected = null;
        WorkerStatus.VitLoad selectedSample = null;
        String address = null;
        long bestLoad = Long.MAX_VALUE;
        int ties = 0;
        long nowNanos = System.nanoTime();
        for (var entry : workers.vitRoutingSnapshot()) {
            WorkerStatus worker = entry.endpoint().getStatus();
            if (group != null && !Objects.equals(group, worker.getGroup())) {
                continue;
            }
            WorkerStatus.VitLoad sample = worker.vitLoad();
            long load = load(sample, nowNanos, config);
            if (load < bestLoad) {
                bestLoad = load;
                ties = 0;
            }
            if (load == bestLoad && ThreadLocalRandom.current().nextInt(++ties) == 0) {
                selected = worker;
                selectedSample = sample;
                address = entry.address();
            }
        }
        ServerStatus result = checkedStatus(selected, address, ctx.getRequestId(), group);
        if (result.isSuccess()) {
            recordLoad(span, selectedSample, nowNanos, config);
        }
        return result;
    }

    private static long load(WorkerStatus.VitLoad sample, long nowNanos, RoutingConfig.VitConfig config) {
        return sample != null && nowNanos - sample.receivedNanos()
                <= TimeUnit.MILLISECONDS.toNanos(config.getLoadMaxAgeMs())
                ? sample.inflightInputs() : Long.MAX_VALUE;
    }

    private ServerStatus selectAffinity(BalanceContext ctx, String group,
            RoutingConfig.VitConfig config, List<String> keys, Map<String, Candidate> candidates, Span span) {
        long now = System.currentTimeMillis();
        long nowNanos = System.nanoTime();
        List<LoadedCandidate> best = new ArrayList<>();
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
            WorkerStatus.VitLoad sample = candidate.worker().vitLoad();
            long load = load(sample, nowNanos, config);
            int comparison = bestScore == null ? 1 : score.compareTo(bestScore);
            if (comparison == 0) { comparison = Long.compare(bestLoad, load); }
            if (comparison > 0) {
                best.clear();
                bestScore = score;
                bestLoad = load;
            }
            if (comparison >= 0) {
                best.add(new LoadedCandidate(candidate, sample));
            }
        }
        if (best.isEmpty()) {
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        LoadedCandidate selectedLoad = best.get(ThreadLocalRandom.current().nextInt(best.size()));
        Candidate selected = selectedLoad.candidate();
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
        recordLoad(span, selectedLoad.sample(), nowNanos, config);
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
        return traceSelection(ctx, "validate", span -> validateImpl(ctx, group, span));
    }

    private ServerStatus validateImpl(BalanceContext ctx, String group, Span span) {
        ServerStatus selected = ctx.getRequest().getSelectedVit();
        if (selected == null) {
            FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_VALIDATION_REASON, "missing_selection");
            return errorStatus(StrategyErrorType.NO_VIT_WORKER);
        }
        try (var pin = workers.captureEndpoint(RoleType.VIT,
                selected.getServerIp() + ":" + selected.getHttpPort())) {
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
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_ADDRESS, selected.getServerIp() + ":" + selected.getGrpcPort());
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_CURRENT_GRPC_PORT, (long) worker.getGrpcPort());
            } else if (!Objects.equals(selected.getGroup(), worker.getGroup())) {
                reason = "selected_group_mismatch";
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_SELECTED_GROUP, selected.getGroup());
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_CURRENT_GROUP, worker.getGroup());
            } else if (selected.getWorkerGeneration() != pin.generationId()) {
                reason = "generation_mismatch";
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_WORKER_GENERATION, selected.getWorkerGeneration());
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_CURRENT_GENERATION, pin.generationId());
            }
            if (!reason.equals("valid")) {
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_VALIDATION_REASON, reason);
                return errorStatus(StrategyErrorType.NO_VIT_WORKER);
            }
            // Validation does not reselect by load: report its current snapshot separately.
            if (span != null) {
                try {
                    RoutingConfig.VitConfig config = ctx.getConfig() == null
                            ? new RoutingConfig.VitConfig() : ctx.getConfig().getRouter().getRoles().getVit();
                    recordLoad(span, worker.vitLoad(), System.nanoTime(), config);
                } catch (RuntimeException ignored) {
                    // Telemetry must not affect endpoint validation.
                }
            }
            return status(worker, ctx.getRequestId());
        }
    }

    private static void recordLoad(Span span, WorkerStatus.VitLoad sample, long nowNanos,
                                   RoutingConfig.VitConfig config) {
        if (span == null) { return; }
        try {
            if (sample == null) {
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_LOAD_STATE, "missing");
                return;
            }
            long ageNanos = nowNanos - sample.receivedNanos();
            FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_MM_INFLIGHT_INPUTS, sample.inflightInputs());
            FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_LOAD_SAMPLE_AGE_MS,
                    TimeUnit.NANOSECONDS.toMillis(Math.max(0, ageNanos)));
            FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_LOAD_STATE,
                    ageNanos <= TimeUnit.MILLISECONDS.toNanos(config.getLoadMaxAgeMs()) ? "fresh" : "stale");
        } catch (RuntimeException ignored) {
            // The unknown-load routing sentinel must never be exported as a media count.
        }
    }

    private ServerStatus traceSelection(BalanceContext ctx, String operation, Function<Span, ServerStatus> action) {
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
                    FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_ROUTE_ONLY, request.isVitRouteOnly());
                    FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_KEY_COUNT,
                            request.getCacheAffinityKeys() == null ? 0L : (long) request.getCacheAffinityKeys().size());
                }
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_CACHE_AFFINITY,
                        ctx.getConfig() != null && ctx.getConfig().getRouter().getRoles().getVit().isCacheAffinityEnabled());
            }
        } catch (RuntimeException ignored) {
            // Observability must not change selector validation or admission.
        }
        try {
            ServerStatus result = action.apply(span);
            FlexlbTrace.setAttribute(span, FlexlbTrace.SCHEDULE_CODE, (long) result.getCode());
            if (result.isSuccess()) {
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_ADDRESS,
                        result.getServerIp() + ":" + result.getGrpcPort());
                FlexlbTrace.setAttribute(span, FlexlbTrace.VIT_WORKER_GENERATION, result.getWorkerGeneration());
            } else {
                FlexlbTrace.markBusinessError(traceContext, result.getCode(),
                        operation.equals("validate") ? "VIT_ROUTE_STALE" : "VIT_SELECTION_REJECTED");
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

    /** Holds the selected endpoint generation through the admission handoff. */
    public SelectedRole selectPinned(BalanceContext ctx, String group) {
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
