package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Red→green minimal reproductions for the plan-commit concurrency redesign
 * (design doc plan_commit_concurrency_redesign.md, N2/N1/N3):
 *
 * <ul>
 *   <li>C-1: queued (not yet dispatched) shadow reservations must not count
 *       against the decode engine concurrency limit — root cause C of the
 *       8400 storm (shadow saturation while the engine is idle).</li>
 *   <li>C-2: an INFEASIBLE decode eviction must fall back to a normal
 *       capacity failure (retry, then a reason-tagged exhaustion) instead of
 *       terminating the request on the first attempt.</li>
 *   <li>A-1: an orphan decode reservation (no scheduler inflight entry) must
 *       be reclaimed by the inflight TTL cleanup pass (P1-4).</li>
 *   <li>B-1: the SLO-deadline rejection must carry a machine-readable
 *       {@code reason=} tag so 8400s are attributable.</li>
 *   <li>D-1: an infeasible eviction must emit its stable metric and preserve
 *       every existing reservation.</li>
 * </ul>
 */
class PlanCommitConcurrencyRedesignTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    private ConfigService configService;
    private Router router;
    private EngineGrpcClient grpcClient;
    private BatchSchedulerReporter reporter;
    private PrioritySchedulerReporter priorityReporter;
    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;
    private WorkerStatus decodeWs;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setAutoTpmEnabled(true);
        when(configService.loadBalanceConfig()).thenReturn(config);

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId());
        });
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    return CompletableFuture.completedFuture(ackFor(request));
                });

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        PriorityAdmissionScheduler priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                new PrioritySloPolicy(PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                        PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel());
        scheduler = new FlexlbBatchScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, priorityScheduler, null);

        WorkerStatus prefillWs = new WorkerStatus();
        prefillWs.setIp("10.0.0.1");
        prefillWs.setPort(8080);
        prefillWs.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, prefillWs);

        decodeWs = new WorkerStatus();
        decodeWs.setIp("10.0.0.2");
        decodeWs.setPort(8081);
        decodeWs.setGrpcPort(8082);
        decodeWs.setAvailableKvCacheTokens(new java.util.concurrent.atomic.AtomicLong(1_000_000L));
        decodeWs.setTotalKvCacheTokens(new java.util.concurrent.atomic.AtomicLong(2_000_000L));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    // ==================== C-1: queued reservations vs engine concurrency ====================

    /**
     * Root cause C: shadow reservations of requests still sitting in the
     * prefill queue must not saturate the decode concurrency limit. Before
     * the fix the availability gate compares {@code getTotalLoad()} (which
     * counts queued reservations) against the limit → the endpoint looks full
     * while the engine is idle.
     */
    @Test
    void c1_queued_reservations_do_not_saturate_engine_concurrency() throws Exception {
        config.setDecodeConcurrencyLimit(4);
        config.setFlexlbBatchSizeMax(100); // batch never fills → items stay queued
        decodeWs.setAlive(true);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        // Route performs the decode reservation (D reserve first), like production
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            decodeEp.reserve(ctx.getRequestId(), 128, 136);
            return successRoute(ctx.getRequestId());
        });

        for (long id = 101; id <= 104; id++) {
            scheduler.submit(context(id));
        }
        awaitQueueSize(4);

        // Shadow accounting still tracks all four queued reservations …
        assertEquals(4, decodeEp.getTotalLoad());
        // … but none of them has been dispatched, so the engine-facing
        // concurrency gate must still report the endpoint as available.
        assertTrue(new DecodeResourceMeasure(configService).isResourceAvailable(decodeEp),
                "queued-only reservations must not close the decode concurrency gate");
    }

    /**
     * P1-1 (task33 review): the queued-phase mark must be set BEFORE the plan
     * commit publishes the item to the batcher. Marking after the commit (the
     * pre-fix onCommitted) races the dispatch-side ownership claim: an
     * item that dispatches immediately is unmarked first and re-marked
     * afterwards, leaving a stale queued mark that hides the dispatched
     * request from the engine concurrency gate until the next calibrate.
     *
     * <p>Deterministic interleaving: {@code reportPlanAge} (which the pre-fix
     * onCommitted invoked right before its late markQueuedPhase) blocks until
     * the gRPC dispatch — which runs after tryMarkEngineMayHaveSeen — happened;
     * {@code reportNormalPlacement} (invoked after the pre-fix late mark)
     * releases the assertion. Pre-fix: stale mark ⇒ engineLoad 0 (red).
     * Post-fix: the mark precedes the commit, dispatch clears it ⇒
     * engineLoad 1 (green).
     */
    @Test
    void p1_1_immediate_dispatch_does_not_leave_stale_queued_mark() throws Exception {
        config.setDecodeConcurrencyLimit(4);
        config.setFlexlbBatchSizeMax(1); // dispatch immediately on offer
        decodeWs.setAlive(true);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        // Route performs the decode reservation (D reserve first), like production
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            decodeEp.reserve(ctx.getRequestId(), 128, 136);
            return successRoute(ctx.getRequestId());
        });

        CountDownLatch dispatched = new CountDownLatch(1);
        CountDownLatch committed = new CountDownLatch(1);
        when(grpcClient.batchEnqueueAsync(anyString(), anyInt(),
                any(EngineRpcService.EnqueueBatchRequestPB.class), anyLong()))
                .thenAnswer(inv -> {
                    // The engine-visibility claim already ran — it precedes dispatch.
                    dispatched.countDown();
                    EngineRpcService.EnqueueBatchRequestPB request = inv.getArgument(2);
                    return CompletableFuture.completedFuture(ackFor(request));
                });
        doAnswer(inv -> {
            // Pre-fix onCommitted marks right after this call — force the
            // late mark to land strictly after the dispatch-side unmark.
            assertTrue(dispatched.await(2, TimeUnit.SECONDS), "dispatch never happened");
            return null;
        }).when(priorityReporter).reportPlanAge(anyInt(), anyLong());
        doAnswer(inv -> {
            committed.countDown(); // any pre-fix late mark is applied by now
            return null;
        }).when(priorityReporter).reportNormalPlacement(anyInt());

        Response response = scheduler.submit(context(105)).get(2, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());
        assertTrue(committed.await(2, TimeUnit.SECONDS), "onCommitted never finished");

        // The reservation is dispatched (not queued): it must count against
        // the engine concurrency gate again.
        assertEquals(1, decodeEp.getTotalLoad());
        assertEquals(1, decodeEp.getEngineLoad(),
                "dispatched reservation hidden by a stale queued-phase mark");
    }

    // ==================== C-2: INFEASIBLE eviction falls back to capacity failure ====================

    /**
     * Root cause C escalation: with decode reserved-only eviction enabled, an
     * INFEASIBLE plan (no strictly-lower-priority candidates) must not be a
     * first-attempt terminal 8400. It should count as a capacity failure,
     * consume the retry budget, and surface a reason-tagged exhaustion.
     */
    @Test
    void c2_infeasible_decode_eviction_retries_and_tags_capacity_reason() throws Exception {
        config.setAutoTpmDecodeReservedEvictEnabled(true);
        config.setDecodeConcurrencyLimit(4);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        // Four same-priority (50) reservations: slotDeficit=1, but no
        // strictly-lower-priority candidate → planDecode is INFEASIBLE.
        for (long id = 801; id <= 804; id++) {
            decodeEp.reserve(id, 128, 136, 50, 0);
        }
        // Router reports a decode-capacity failure (8403) → Phase 4 eviction path
        when(router.route(any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_DECODE_WORKER));

        Response response = scheduler.submit(context(200)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                response.getAdmissionRejectReason());
        verify(router, times(1)).route(any(BalanceContext.class));
    }

    // ==================== A-1: orphan decode reservation reclaimed by cleanup ====================

    /**
     * P1-4: a decode shadow reservation without a matching scheduler inflight
     * entry (e.g. interrupted between route() and registerInflight) must be
     * reclaimed by the periodic inflight cleanup once past the TTL; a miss on
     * finishYieldedById must stay a harmless no-op.
     */
    @Test
    void a1_orphan_decode_reservation_is_reclaimed_by_cleanup() throws Exception {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        decodeEp.reserve(888, 128, 136); // orphan: never inflight-registered
        assertEquals(1, decodeEp.getInflightCount());

        config.setFlexlbInflightTtlMs(0);
        Thread.sleep(10); // ensure the orphan is past the (zero) TTL
        scheduler.cleanupInflight();

        // finishYieldedById on an unknown id must remain a no-op either way
        scheduler.finishYieldedById(888, "stale victim settle");

        assertEquals(0, decodeEp.getInflightCount(),
                "orphan reservation must be reclaimed by cleanupInflight");
        assertEquals(0, decodeEp.inflightHardKvReserved());
    }

    // ==================== B-1: SLO rejection carries a reason tag ====================

    @Test
    void b1_slo_deadline_rejection_is_typed_resource_exhaustion() throws Exception {
        BalanceContext ctx = context(300);
        ctx.setBudget(ScheduleBudget.forDeadline(50,
                ctx.getStartTime(), System.currentTimeMillis() - 1_000));

        Response response = scheduler.submit(ctx).get(1, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED,
                response.getAdmissionRejectReason());
        assertTrue(response.getErrorMessage().contains("admission budget already expired"));
    }

    // ==================== D-1: infeasible plan has metric + no side effects ====================

    @Test
    void d1_infeasible_plan_reports_metric_and_preserves_reservations() throws Exception {
        config.setAutoTpmDecodeReservedEvictEnabled(true);
        config.setDecodeConcurrencyLimit(4);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        for (long id = 811; id <= 814; id++) {
            decodeEp.reserve(id, 128, 136, 50, 0);
        }
        when(router.route(any(BalanceContext.class)))
                .thenReturn(Response.error(StrategyErrorType.NO_DECODE_WORKER));

        Response response = scheduler.submit(context(400)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(AdmissionRejectReason.SAME_PRIORITY_AHEAD,
                response.getAdmissionRejectReason());
        verify(priorityReporter).reportEvictionPlan(
                50, "decode_slot_and_kv_full", "infeasible");
        verify(priorityReporter, never()).reportEvictionCommit(
                anyInt(), anyString(), anyString());
        assertEquals(4, decodeEp.getInflightCount());
        for (long id = 811; id <= 814; id++) {
            assertTrue(decodeEp.reservedView().containsKey(id));
        }
    }

    // ==================== N3: lockfree commit & presence guard ====================

    /**
     * N3 §3.3: the default lockfree commit must not abort on unrelated
     * prefill queue-version churn between the plan snapshot and the commit.
     * Under the legacy versioned strategy this exact interference costs a
     * VERSION_MISMATCH plus a full re-route (85%+ commit failures at
     * production QPS — the 8515 storm).
     */
    @Test
    void n3_lockfree_commit_ignores_queue_version_drift() throws Exception {
        config.setFlexlbBatchSizeMax(100); // items stay queued, no dispatch
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();

        AtomicInteger routeCalls = new AtomicInteger();
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            if (routeCalls.incrementAndGet() == 1) {
                // Concurrent enqueue between snapshot and commit → version bump
                assertTrue(batcher.tryOffer(dummyItem(901)));
            }
            endpointRegistry.getDecode(DECODE_IP_PORT)
                    .reserve(ctx.getRequestId(), 128, 136,
                            ctx.getPriority(), ctx.getDeadlineMs());
            return successRoute(ctx.getRequestId());
        });

        Response response = scheduler.submit(context(501)).get(2, TimeUnit.SECONDS);

        assertTrue(response.isSuccess());
        // Single attempt: no VERSION_MISMATCH abort, no re-route
        verify(router, times(1)).route(any(BalanceContext.class));
        verify(priorityReporter, never()).reportPlanConflict("normal_placement_version");
        // N3 observability: committed plan age is reported
        verify(priorityReporter).reportPlanAge(eq(50), anyLong());
    }

    /**
     * N3 §3.3 retry shrink: on the lockfree path a capacity failure is not a
     * transient conflict — after the primary offer and one fallback re-route
     * both hit OFFER_FAILED, reject fast with a reason tag instead of burning
     * the full re-route budget.
     */
    @Test
    void n3_lockfree_fast_rejects_after_primary_and_one_fallback_offer() throws Exception {
        config.setFlexlbBatchQueueMaxSize(1);
        config.setFlexlbBatchSizeMax(100);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        // Route performs the decode reservation (D reserve first), like production
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            decodeEp.reserve(ctx.getRequestId(), 128, 136);
            return successRoute(ctx.getRequestId());
        });
        // Fill the single queue slot so every tryOffer() fails
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        assertTrue(batcher.tryOffer(dummyItem(999)));

        Response response = scheduler.submit(context(502)).get(2, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        // The queue is occupied by a legacy item created without an Auto-TPM
        // ScheduleBudget, so its priority is the untrusted 0 sentinel.  The
        // fast-reject timing remains N3's contract, but the causal result
        // cannot truthfully claim typed priority or pure resource exhaustion.
        assertEquals(StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.UNSPECIFIED,
                response.getAdmissionRejectReason());
        // Primary + one fallback re-route — not the full 3-attempt budget
        verify(router, times(2)).route(any(BalanceContext.class));
        // Rollback: every decode reservation released
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
    }

    /**
     * N3 §3.4: the victim-presence replace commits as long as the victims are
     * still queued — the same unrelated version churn that aborts the legacy
     * queue_version guard no longer matters.
     */
    @Test
    void n3_presence_replace_survives_version_drift_where_versioned_guard_aborts() {
        config.setFlexlbBatchSizeMax(100);
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        PrefillQueueManager queueManager = batcher.queueManager();
        assertTrue(batcher.tryOffer(dummyItem(601))); // victim
        long staleVersion = queueManager.queueVersion();
        assertTrue(batcher.tryOffer(dummyItem(602))); // unrelated churn → version bump

        // Red (legacy guard): unrelated churn aborts the commit …
        assertTrue(queueManager.tryReplaceVictimsWithIncoming(
                List.of(601L), dummyItem(611), staleVersion).isVersionMismatch());
        // … green (presence guard): the victim is still queued → replace lands
        PrefillQueueManager.ReplaceOutcome outcome =
                queueManager.tryReplaceVictimsPresent(List.of(601L), dummyItem(612));
        assertTrue(outcome.isSuccess());
        assertEquals(601L, outcome.removed().get(0).requestId());
    }

    /**
     * N3 §3.4: a victim missing from the queue aborts the presence replace
     * with zero side effects — nothing removed, incoming not enqueued, and
     * the missing ids are surfaced for the victim-gone handling.
     */
    @Test
    void n3_presence_replace_victim_gone_is_zero_side_effect() {
        config.setFlexlbBatchSizeMax(100);
        WorkerBatcher batcher = endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher();
        PrefillQueueManager queueManager = batcher.queueManager();
        assertTrue(batcher.tryOffer(dummyItem(701)));
        int depthBefore = batcher.queueSize();

        PrefillQueueManager.ReplaceOutcome outcome =
                queueManager.tryReplaceVictimsPresent(List.of(777L), dummyItem(702));

        assertTrue(outcome.isVictimGone());
        assertEquals(List.of(777L), outcome.missingVictimIds());
        assertTrue(outcome.removed().isEmpty());
        assertEquals(depthBefore, batcher.queueSize());
    }

    /**
     * N3 §3.4: {@code releaseIfHeld} is a CAS-style conditional release —
     * it frees a still-held RESERVED_NOT_ACCEPTED entry exactly once and is
     * a no-op {@code false} when the reservation is already gone.
     */
    @Test
    void n3_release_if_held_is_cas_style_conditional() {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        decodeEp.reserve(31, 128, 136);
        decodeEp.markQueuedPhase(31);

        assertTrue(decodeEp.releaseIfHeld(31));
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
        // Reservation gone → conditional release must not double-release
        assertFalse(decodeEp.releaseIfHeld(31));
    }

    /**
     * N3 §3.4: when only part of the decode victims are still held, the
     * presence eviction keeps the freed releases (their host requests are
     * driven terminal by the caller — no rollback) and reports a replan
     * without reserving the incoming.
     */
    @Test
    void n3_presence_decode_eviction_partial_release_keeps_freed_and_replans() {
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        decodeEp.reserve(41, 128, 136, 30, 0);
        decodeEp.markQueuedPhase(41);
        // Victim 42 no longer holds a reservation (already dispatched/settled)

        DecodeEndpoint.PresenceEvictionOutcome outcome =
                decodeEp.tryReleaseVictimsIfHeldAndReserveIncoming(
                        List.of(41L, 42L), 900, 128, 136, 70, 0);

        assertFalse(outcome.success());
        assertEquals(List.of(41L), outcome.freedVictimIds());
        // Freed victim stays released (no rollback), incoming not reserved
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
    }

    // ==================== helpers ====================

    private BatchItem dummyItem(long requestId) {
        Response route = successRoute(requestId);
        return new BatchItem(context(requestId), new CompletableFuture<>(), route,
                FlexlbBatchScheduler.findServer(route, RoleType.PREFILL),
                FlexlbBatchScheduler.findServer(route, RoleType.DECODE),
                endpointRegistry.getPrefill(PREFILL_IP_PORT), null,
                System.currentTimeMillis());
    }

    /** Wait (max 2s) until the prefill queue holds the expected item count. */
    private void awaitQueueSize(int expected) throws InterruptedException {
        long deadline = System.currentTimeMillis() + 2_000;
        while (System.currentTimeMillis() < deadline) {
            if (endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher().queueSize() == expected) {
                return;
            }
            Thread.sleep(10);
        }
        assertEquals(expected,
                endpointRegistry.getPrefill(PREFILL_IP_PORT).getBatcher().queueSize());
    }

    private static EngineRpcService.EnqueueBatchResponsePB ackFor(
            EngineRpcService.EnqueueBatchRequestPB request) {
        EngineRpcService.EnqueueBatchResponsePB.Builder response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(request.getBatchId());
        request.getDpSlotsList().stream()
                .flatMap(slot -> slot.getRequestsList().stream())
                .map(EngineRpcService.EnqueueBatchExternalInputPB::getInput)
                .forEach(input -> response.addSuccesses(
                        EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                .setRequestId(input.getRequestId())
                                .build()));
        return response.build();
    }

    private static BalanceContext context(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(50);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        ctx.setGenerateInputPbBytes(generateInputBytes(requestId));
        return ctx;
    }

    private static byte[] generateInputBytes(long requestId) {
        EngineRpcService.GenerateInputPB input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .addTokenIds(101)
                .addTokenIds(102)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .build())
                .build();
        return input.toByteArray();
    }

    private static Response successRoute(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, "10.0.0.1", 8080, 8081, requestId),
                server(RoleType.DECODE, "10.0.0.2", 8081, 8082, requestId)
        ));
        return response;
    }

    private static ServerStatus server(RoleType role, String ip, int httpPort, int grpcPort, long requestId) {
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(role);
        status.setServerIp(ip);
        status.setHttpPort(httpPort);
        status.setGrpcPort(grpcPort);
        status.setDpRank(0);
        status.setGroup("g1");
        status.setRequestId(requestId);
        return status;
    }
}
