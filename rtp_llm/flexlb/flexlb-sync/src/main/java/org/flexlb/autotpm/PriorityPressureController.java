package org.flexlb.autotpm;

import io.grpc.Status;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.util.Logger;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;

/**
 * Auto-TPM running-decode preemption orchestrator (decision D6) — the
 * admission layer BatchScheduler consults when routing fails for lack of
 * decode capacity.
 *
 * <p>Flow (blueprint §1.7): master switch → candidate snapshot → victim
 * selection → rate-limit guardrail → engine Cancel RPC (reason =
 * PRIORITY_PREEMPTED) → response interpretation → bounded release wait →
 * victim settlement — strictly in that order, and never optimistic:
 * <ul>
 *   <li>{@code found=false} — the victim raced to completion; the permit is
 *       rolled back and no capacity is assumed freed</li>
 *   <li>{@code found=true} — bounded wait (default 50ms, 5ms polls) for the
 *       victim's layer-2 entry to disappear (calibrate close-out signal)</li>
 *   <li>wait timeout — the incoming request is NOT dispatched; the victim
 *       keeps its cancel intent and the WorkerStatus close-out path
 *       ({@code DecodeEndpoint#processFinishedTasks}) settles it later</li>
 *   <li>release confirmed — the victim's {@link InflightItem} is completed
 *       as {@link StrategyErrorType#AUTO_TPM_PREEMPTED} (idempotent CAS) and
 *       a {@link PreemptResult} is returned</li>
 * </ul>
 *
 * <p>Forbidden by design (predecessor pitfall 2 — KV double-count): firing
 * the cancel and immediately dropping local accounting. Local layer-2 state
 * is only cleared by calibrate observing the engine's finished report.
 *
 * <p>Note on guardrail ordering: the blueprint lists
 * {@code rateLimiter.tryAcquire} before candidate collection, but the
 * limiter's per-endpoint window needs the victim's endpoint, so the permit
 * is taken right after victim selection — no permit is burned when there is
 * no eligible victim, and the guardrail still gates every Cancel RPC.
 */
public class PriorityPressureController {

    // running_cancel.count result tag values (D10)
    public static final String RESULT_SUCCESS = "success";
    public static final String RESULT_TIMEOUT = "timeout";
    public static final String RESULT_NOT_FOUND = "not_found";
    public static final String RESULT_UNSUPPORTED = "unsupported";
    public static final String RESULT_RATE_LIMITED = "rate_limited";

    /** Poll interval of the bounded release wait. */
    private static final long RELEASE_POLL_INTERVAL_MS = 5;

    /**
     * Safety TTL for cancel intents whose release wait timed out: keeps the
     * victim shielded from double selection while the WorkerStatus close-out
     * is pending, without leaking the entry forever if the report never comes
     * (stale-round eviction settles the item independently).
     */
    private static final long CANCEL_INTENT_TTL_MS = 5_000;

    private final ConfigService configService;
    private final EndpointRegistry endpointRegistry;
    private final EngineGrpcClient grpcClient;
    private final InflightStore inflightStore;
    private final PriorityRegistry priorityRegistry;
    private final FlexlbMetricHelper metricHelper;

    /** requestId → intent timestamp; present while a cancel is in flight. */
    private final ConcurrentHashMap<Long, Long> cancelIntents = new ConcurrentHashMap<>();

    /** Limiter rebuilt lazily when the configured limits change. */
    private volatile PreemptRateLimiter rateLimiter;
    private volatile long rateLimiterKey = Long.MIN_VALUE;

    public PriorityPressureController(ConfigService configService,
                                      EndpointRegistry endpointRegistry,
                                      EngineGrpcClient grpcClient,
                                      InflightStore inflightStore,
                                      PriorityRegistry priorityRegistry,
                                      FlexlbMetricHelper metricHelper) {
        this.configService = configService;
        this.endpointRegistry = endpointRegistry;
        this.grpcClient = grpcClient;
        this.inflightStore = inflightStore;
        this.priorityRegistry = priorityRegistry;
        this.metricHelper = metricHelper;
    }

    /**
     * Try to free decode capacity for {@code ctx} by preempting one strictly
     * lower-priority RUNNING request.
     *
     * @return the freed endpoint on confirmed release; empty on any
     *         non-confirmed outcome (switch off, no victim, rate-limited,
     *         race, degrade, timeout) — the caller must NOT dispatch
     *         optimistically
     */
    public Optional<PreemptResult> tryPreempt(BalanceContext ctx) {
        FlexlbConfig config = configService.loadBalanceConfig();
        if (config == null || !config.isAutoTpmEnabled() || !config.isAutoTpmDecodeRunningPreemptEnabled()) {
            return Optional.empty();
        }
        pruneExpiredIntents();

        int incomingPriority = ctx.getPriority();
        long requestId = ctx.getRequestId();
        long now = System.currentTimeMillis();

        List<VictimCandidate> candidates = new ArrayList<>();
        for (Map.Entry<String, DecodeEndpoint> entry : endpointRegistry.getDecodeEndpoints().entrySet()) {
            candidates.addAll(entry.getValue().snapshotRunningCandidates(priorityRegistry));
        }
        Optional<VictimCandidate> selected = InflightVictimSelector.select(candidates, incomingPriority,
                config.getAutoTpmPreemptCriticalSectionMs(), now, cancelIntents::containsKey);
        if (selected.isEmpty()) {
            return Optional.empty();
        }
        VictimCandidate victim = selected.get();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(victim.endpoint());
        if (decodeEp == null) {
            return Optional.empty();
        }

        PreemptRateLimiter limiter = rateLimiter(config);
        if (!limiter.tryAcquire(victim.endpoint())) {
            reportCancel(victim.priority(), incomingPriority, RESULT_RATE_LIMITED);
            Logger.info("FlexLB preempt_rate_limited request_id={} priority={} victim={} victim_priority={}",
                    requestId, incomingPriority, victim.requestId(), victim.priority());
            return Optional.empty();
        }
        metricHelper.reportAutoTpmPreemptRate(limiter.globalCount());

        if (!grpcClient.isCancelSupported(decodeEp.getIp(), decodeEp.getGrpcPort())) {
            limiter.rollback(victim.endpoint());
            reportCancel(victim.priority(), incomingPriority, RESULT_UNSUPPORTED);
            return Optional.empty();
        }

        long waitTimeoutMs = Math.max(1, config.getAutoTpmCommitWaitReleaseTimeoutMs());
        cancelIntents.put(victim.requestId(), now);
        Logger.info("FlexLB preempt_cancel request_id={} priority={} victim={} victim_priority={} endpoint={}",
                requestId, incomingPriority, victim.requestId(), victim.priority(), victim.endpoint());

        EngineRpcService.CancelResponsePB cancelResponse;
        try {
            cancelResponse = grpcClient
                    .cancelAsync(decodeEp.getIp(), decodeEp.getGrpcPort(), victim.requestId(),
                            EngineRpcService.EngineCancelReasonPB.ENGINE_CANCEL_REASON_PRIORITY_PREEMPTED,
                            waitTimeoutMs)
                    .get(waitTimeoutMs + RELEASE_POLL_INTERVAL_MS, TimeUnit.MILLISECONDS);
        } catch (Exception e) {
            return handleCancelFailure(e, victim, incomingPriority, requestId, limiter);
        }

        if (!cancelResponse.getFound() && !cancelResponse.getAlreadyFinished()) {
            // Race: the victim left the engine between snapshot and cancel.
            // Not optimistic — no capacity is assumed freed, the permit goes back.
            cancelIntents.remove(victim.requestId());
            limiter.rollback(victim.endpoint());
            reportCancel(victim.priority(), incomingPriority, RESULT_NOT_FOUND);
            Logger.info("FlexLB preempt_not_found request_id={} priority={} victim={}",
                    requestId, incomingPriority, victim.requestId());
            return Optional.empty();
        }

        // found=true (engine is aborting the stream) or already_finished
        // (finished report imminent): both must still be CONFIRMED locally by
        // the layer-2 entry disappearing before any capacity is granted.
        boolean released = awaitRelease(decodeEp, victim.requestId(), waitTimeoutMs);
        if (!released) {
            // Not optimistic: keep the cancel intent; the WorkerStatus
            // close-out path settles the victim when the report arrives.
            reportCancel(victim.priority(), incomingPriority, RESULT_TIMEOUT);
            Logger.warn("FlexLB preempt_wait_timeout request_id={} priority={} victim={} timeout_ms={}",
                    requestId, incomingPriority, victim.requestId(), waitTimeoutMs);
            return Optional.empty();
        }

        if (cancelResponse.getFound()) {
            // Cancelled by us → structured attribution 4290. already_finished
            // victims completed on their own and keep their natural terminal.
            settleVictim(victim.requestId());
        }
        cancelIntents.remove(victim.requestId());
        reportCancel(victim.priority(), incomingPriority, RESULT_SUCCESS);
        Logger.info("FlexLB preempt_confirmed request_id={} priority={} victim={} endpoint={}",
                requestId, incomingPriority, victim.requestId(), victim.endpoint());
        return Optional.of(new PreemptResult(victim.endpoint(), victim.requestId(), victim.priority()));
    }

    /** Whether a cancel intent is pending for {@code requestId} (test/selector visibility). */
    public boolean hasCancelIntent(long requestId) {
        return cancelIntents.containsKey(requestId);
    }

    // ==================== internals ====================

    private Optional<PreemptResult> handleCancelFailure(Exception e,
                                                        VictimCandidate victim,
                                                        int incomingPriority,
                                                        long requestId,
                                                        PreemptRateLimiter limiter) {
        Throwable cause = e instanceof ExecutionException ? e.getCause() : e;
        Status.Code code = Status.fromThrowable(cause).getCode();
        if (code == Status.Code.UNIMPLEMENTED) {
            // Old engine — EngineGrpcClient has already degraded the endpoint.
            cancelIntents.remove(victim.requestId());
            limiter.rollback(victim.endpoint());
            reportCancel(victim.priority(), incomingPriority, RESULT_UNSUPPORTED);
            Logger.warn("FlexLB preempt_unsupported request_id={} victim={} endpoint={}",
                    requestId, victim.requestId(), victim.endpoint());
            return Optional.empty();
        }
        if (code == Status.Code.DEADLINE_EXCEEDED || e instanceof java.util.concurrent.TimeoutException) {
            // The cancel may still land engine-side: keep the intent (and the
            // permit) so WorkerStatus close-out can settle the victim.
            reportCancel(victim.priority(), incomingPriority, RESULT_TIMEOUT);
            Logger.warn("FlexLB preempt_cancel_rpc_timeout request_id={} victim={} endpoint={}",
                    requestId, victim.requestId(), victim.endpoint());
            return Optional.empty();
        }
        // Transport-level failure — the cancel did not reach the engine.
        cancelIntents.remove(victim.requestId());
        limiter.rollback(victim.endpoint());
        reportCancel(victim.priority(), incomingPriority, RESULT_NOT_FOUND);
        Logger.warn("FlexLB preempt_cancel_failed request_id={} victim={} endpoint={} err={}",
                requestId, victim.requestId(), victim.endpoint(), cause == null ? null : cause.getMessage());
        return Optional.empty();
    }

    /**
     * Bounded wait for the victim's layer-2 entry to disappear — the
     * calibrate-driven close-out signal that the engine really released the
     * capacity. Polls every {@link #RELEASE_POLL_INTERVAL_MS}.
     */
    private boolean awaitRelease(DecodeEndpoint decodeEp, long victimRequestId, long timeoutMs) {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (true) {
            if (!decodeEp.hasEngineTask(victimRequestId)) {
                return true;
            }
            long remaining = deadline - System.currentTimeMillis();
            if (remaining <= 0) {
                return false;
            }
            try {
                Thread.sleep(Math.min(RELEASE_POLL_INTERVAL_MS, remaining));
            } catch (InterruptedException ie) {
                Thread.currentThread().interrupt();
                return !decodeEp.hasEngineTask(victimRequestId);
            }
        }
    }

    /**
     * Settle the preempted victim's {@link InflightItem} with
     * {@link StrategyErrorType#AUTO_TPM_PREEMPTED} (4290). Idempotent: the
     * CAS-guarded terminal transition makes a duplicate settle (e.g. the
     * WorkerStatus close-out racing this call) a no-op.
     */
    private void settleVictim(long victimRequestId) {
        InflightItem item = inflightStore.get(String.valueOf(victimRequestId));
        if (item != null && !item.isTerminated()) {
            item.complete(Response.error(StrategyErrorType.AUTO_TPM_PREEMPTED));
        }
    }

    private void reportCancel(int victimPriority, int incomingPriority, String result) {
        metricHelper.reportAutoTpmRunningCancel(victimPriority, incomingPriority, result);
    }

    private void pruneExpiredIntents() {
        if (cancelIntents.isEmpty()) {
            return;
        }
        long cutoff = System.currentTimeMillis() - CANCEL_INTENT_TTL_MS;
        cancelIntents.entrySet().removeIf(entry -> entry.getValue() < cutoff);
    }

    private PreemptRateLimiter rateLimiter(FlexlbConfig config) {
        long key = ((long) config.getAutoTpmPreemptRateLimitPerMin() << 32)
                | (config.getAutoTpmEndpointPreemptQpsLimit() & 0xFFFFFFFFL);
        PreemptRateLimiter current = this.rateLimiter;
        if (current == null || key != rateLimiterKey) {
            current = new PreemptRateLimiter(config.getAutoTpmPreemptRateLimitPerMin(),
                    config.getAutoTpmEndpointPreemptQpsLimit());
            this.rateLimiter = current;
            this.rateLimiterKey = key;
        }
        return current;
    }
}
