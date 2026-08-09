package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.BatchDispatcher;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.balance.scheduler.Router;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PrioritySloPolicy;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.PrioritySchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Phase 6 tests for {@link DeadlineRescueService}: danger-zone boundary,
 * priority/transfer-count filters, per-tick and per-endpoint storm limits,
 * CAS-skip isolation, full rescue success (SLO preserved), requeue failure
 * semantics and scanner lifecycle gating.
 */
class DeadlineRescueServiceTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";
    private static final String PREFILL2_IP_PORT = "10.0.0.3:8080";
    private static final String DECODE_IP_PORT = "10.0.0.2:8081";

    private ConfigService configService;
    private Router router;
    private BatchSchedulerReporter reporter;
    private PrioritySchedulerReporter priorityReporter;
    private PriorityAdmissionScheduler priorityScheduler;
    private FlexlbBatchScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private FlexlbConfig config;
    private DeadlineRescueService service;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        router = mock(Router.class);
        EngineGrpcClient grpcClient = mock(EngineGrpcClient.class);
        reporter = mock(BatchSchedulerReporter.class);
        priorityReporter = mock(PrioritySchedulerReporter.class);

        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        // Large batch size + window: queued items are never dispatched, so
        // every tick observes a stable queue.
        config.setFlexlbBatchSizeMax(100);
        config.setFlexlbBatchWindowMs(10_000);
        config.setCostSloMs(50000L);
        config.setCostSloRiskMarginMs(50L);
        config.setAutoTpmEnabled(true);
        config.setAutoTpmDeadlineRescueEnabled(true);
        when(configService.loadBalanceConfig()).thenReturn(config);

        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId(), "10.0.0.1");
        });

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        BatchDispatcher dispatcher = new DefaultBatchDispatcher(grpcClient, configService, null);
        priorityScheduler = new PriorityAdmissionScheduler(
                configService, router, endpointRegistry, new PlanCommitter(),
                new PrioritySloPolicy(PrioritySloPolicy.DEFAULT_SLO_LENGTH_BUCKETS,
                        PrioritySloPolicy.DEFAULT_PRIORITY_SLO_MULTIPLIERS),
                priorityReporter, reporter, new UnsupportedEngineCancelChannel());
        scheduler = new FlexlbBatchScheduler(configService, router,
                endpointRegistry, dispatcher, reporter, priorityScheduler, null);
        service = new DeadlineRescueService(configService, endpointRegistry,
                priorityScheduler, scheduler, priorityReporter);

        registerPrefill(PREFILL_IP_PORT, "10.0.0.1");
        registerPrefill(PREFILL2_IP_PORT, "10.0.0.3");

        WorkerStatus decodeWs = new WorkerStatus();
        decodeWs.setIp("10.0.0.2");
        decodeWs.setPort(8081);
        decodeWs.setGrpcPort(8082);
        decodeWs.setAvailableKvCacheTokens(new AtomicLong(1_000_000L));
        decodeWs.setTotalKvCacheTokens(new AtomicLong(2_000_000L));
        endpointRegistry.ensureEndpoint(RoleType.DECODE, DECODE_IP_PORT, decodeWs);
    }

    @AfterEach
    void tearDown() {
        service.stop();
        scheduler.shutdown();
    }

    // ==================== danger-zone boundary ====================

    @Test
    void rescues_request_exactly_at_danger_threshold() {
        long now = System.currentTimeMillis();
        // remaining == dangerThresholdMs (default 100) → inside the danger zone
        BatchItem item = enqueueDanger(1, 70, now + 100, PREFILL_IP_PORT);

        int migrated = service.rescueTick(now);

        assertEquals(1, migrated);
        verify(priorityReporter).reportRescue(eq(70), eq("success"));
        verify(priorityReporter).reportTransfer(eq(70), eq("success"));
        // §14.4 auxiliary latency metric: from = source endpoint, to = new
        // placement (same endpoint here — the mock route re-selects it).
        verify(priorityReporter).reportRescueLatency(eq(70), eq("success"),
                eq(PREFILL_IP_PORT), eq(PREFILL_IP_PORT), anyLong());
        assertEquals(1, item.ctx().getTransferCount());
    }

    @Test
    void leaves_requests_outside_danger_zone_untouched() {
        long now = System.currentTimeMillis();
        // remaining == threshold + 1 → just outside the danger zone
        enqueueDanger(2, 70, now + 101, PREFILL_IP_PORT);
        // deadline unset (0) → never a candidate
        enqueueDanger(3, 70, 0, PREFILL_IP_PORT);

        int migrated = service.rescueTick(now);

        assertEquals(0, migrated);
        assertEquals(2, queueSize(PREFILL_IP_PORT));
        verify(priorityReporter, never()).reportRescue(anyInt(), anyString());
        verify(priorityReporter, never()).reportTransfer(anyInt(), anyString());
        verify(priorityReporter, never()).reportRescueLatency(
                anyInt(), anyString(), anyString(), anyString(), anyLong());
    }

    @Test
    void rescues_danger_zone_p60_request_when_no_limits_hit() {
        // Design doc §20 Phase 6 acceptance #1: with no per-tick, per-endpoint
        // or transfer-count limit in play, a danger-zone P60 request migrates.
        long now = System.currentTimeMillis();
        BatchItem item = enqueueDanger(14, 60, now + 50, PREFILL_IP_PORT);

        int migrated = service.rescueTick(now);

        assertEquals(1, migrated);
        assertEquals(1, item.ctx().getTransferCount());
        verify(priorityReporter).reportRescue(eq(60), eq("success"));
        verify(priorityReporter).reportTransfer(eq(60), eq("success"));
        verify(priorityReporter, never()).reportRescue(anyInt(), eq("limited"));
        verify(priorityReporter, never()).reportRescue(anyInt(), eq("cas_skipped"));
    }

    // ==================== priority / transfer-count filters ====================

    @Test
    void never_rescues_lowest_priority_request() {
        long now = System.currentTimeMillis();
        enqueueDanger(4, 30, now - 500, PREFILL_IP_PORT); // deep in the danger zone

        int migrated = service.rescueTick(now);

        assertEquals(0, migrated);
        assertEquals(1, queueSize(PREFILL_IP_PORT));
        verify(priorityReporter, never()).reportRescue(anyInt(), anyString());
    }

    @Test
    void skips_request_at_transfer_count_cap() {
        long now = System.currentTimeMillis();
        BatchItem item = enqueueDanger(5, 70, now + 50, PREFILL_IP_PORT);
        item.setTransferCount(1); // == autoTpmMaxTransferCount default (1)

        int migrated = service.rescueTick(now);

        assertEquals(0, migrated);
        assertEquals(1, queueSize(PREFILL_IP_PORT));
        verify(priorityReporter, never()).reportRescue(anyInt(), anyString());
    }

    // ==================== storm limits ====================

    @Test
    void per_tick_limit_migrates_highest_priority_first_and_reports_limited() {
        config.setAutoTpmMaxRescuePerTick(1);
        long now = System.currentTimeMillis();
        enqueueDanger(6, 60, now + 10, PREFILL_IP_PORT);
        enqueueDanger(7, 70, now + 50, PREFILL2_IP_PORT);

        int migrated = service.rescueTick(now);

        // P70 wins (priority desc) even though P60's deadline is closer
        assertEquals(1, migrated);
        verify(priorityReporter).reportTransfer(eq(70), eq("success"));
        verify(priorityReporter).reportRescue(eq(60), eq("limited"));
        verify(priorityReporter, never()).reportTransfer(eq(60), anyString());
        assertEquals(0, queueSize(PREFILL2_IP_PORT)); // P70 migrated away
        assertEquals(2, queueSize(PREFILL_IP_PORT)); // P60 stays + P70 re-placed here
    }

    @Test
    void per_endpoint_limit_caps_migrations_from_one_source_endpoint() {
        config.setAutoTpmMaxRescuePerEndpointPerTick(1);
        long now = System.currentTimeMillis();
        enqueueDanger(8, 70, now + 10, PREFILL_IP_PORT);
        enqueueDanger(9, 60, now + 10, PREFILL_IP_PORT);

        int migrated = service.rescueTick(now);

        assertEquals(1, migrated);
        verify(priorityReporter).reportTransfer(eq(70), eq("success"));
        verify(priorityReporter).reportRescue(eq(60), eq("limited"));
        verify(priorityReporter, never()).reportTransfer(eq(60), anyString());
    }

    // ==================== CAS skip isolation ====================

    @Test
    void cas_failure_skips_candidate_without_affecting_later_ones() {
        long now = System.currentTimeMillis();
        enqueueDanger(10, 70, now + 10, PREFILL_IP_PORT);
        enqueueDanger(11, 60, now + 10, PREFILL_IP_PORT);

        // Interfere once, right before the first candidate's CAS removal:
        // a concurrent enqueue bumps the queue version.
        AtomicBoolean interfered = new AtomicBoolean();
        DeadlineRescueService racy = new DeadlineRescueService(configService,
                endpointRegistry, priorityScheduler, scheduler, priorityReporter) {
            @Override
            protected void onCandidateSelected(long requestId) {
                if (interfered.compareAndSet(false, true)) {
                    BatchItem safe = item(999, 50, 0, PREFILL_IP_PORT);
                    assertTrue(prefill(PREFILL_IP_PORT).getBatcher().tryOffer(safe));
                }
            }
        };

        int migrated = racy.rescueTick(now);

        // First candidate (P70) hit the stale version → cas_skipped; the
        // refreshed version let the second candidate (P60) migrate normally.
        assertEquals(1, migrated);
        verify(priorityReporter).reportRescue(eq(70), eq("cas_skipped"));
        verify(priorityReporter, never()).reportTransfer(eq(70), anyString());
        verify(priorityReporter).reportTransfer(eq(60), eq("success"));
    }

    // ==================== full rescue success ====================

    @Test
    void successful_rescue_migrates_to_new_endpoint_preserving_slo() {
        long now = System.currentTimeMillis();
        long deadline = now + 50;
        BatchItem item = enqueueDanger(12, 70, deadline, PREFILL_IP_PORT);
        long arrival = item.ctx().getStartTime();
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);
        assertEquals(1, decodeEp.getInflightCount());

        // The re-entry route lands on the second (better) prefill endpoint
        when(router.route(any(BalanceContext.class))).thenAnswer(inv -> {
            BalanceContext ctx = inv.getArgument(0);
            return successRoute(ctx.getRequestId(), "10.0.0.3");
        });

        int migrated = service.rescueTick(now);

        assertEquals(1, migrated);
        // Removed from the old queue, placed on the new endpoint's queue
        assertEquals(0, queueSize(PREFILL_IP_PORT));
        assertEquals(1, queueSize(PREFILL2_IP_PORT));
        // Old decode reservation rolled back (mock router does not re-reserve)
        assertEquals(0, decodeEp.getInflightCount());
        assertEquals(0, decodeEp.inflightHardKvReserved());
        // SLO not reset: same deadline/arrival, transferCount bumped to 1
        QueuedRequestSnapshot rescued =
                prefill(PREFILL2_IP_PORT).getBatcher().queueManager().snapshot().items().get(0);
        assertEquals(12, rescued.requestId());
        assertEquals(deadline, rescued.deadlineMs());
        assertEquals(1, rescued.transferCount());
        assertEquals(arrival, item.ctx().getStartTime());
        assertEquals(1, item.ctx().getTransferCount());
        // The original future is still owned by the dispatch pipeline
        assertFalse(item.future().isDone());
        verify(priorityReporter).reportRescue(eq(70), eq("success"));
        verify(priorityReporter).reportTransfer(eq(70), eq("success"));
        // §14.4 latency metric carries both endpoint dimensions
        verify(priorityReporter).reportRescueLatency(eq(70), eq("success"),
                eq(PREFILL_IP_PORT), eq(PREFILL2_IP_PORT), anyLong());
    }

    // ==================== requeue failure ====================

    @Test
    void requeue_failure_completes_original_future_explicitly_without_leak() throws Exception {
        long now = System.currentTimeMillis();
        BatchItem item = enqueueDanger(13, 70, now + 50, PREFILL_IP_PORT);
        DecodeEndpoint decodeEp = endpointRegistry.getDecode(DECODE_IP_PORT);

        when(router.route(any(BalanceContext.class))).thenReturn(null);

        int migrated = service.rescueTick(now);

        assertEquals(1, migrated);
        // The re-entry failed → the original future gets an explicit error
        Response response = item.future().get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), response.getCode());
        // No re-queue, no retry, decode reservation rolled back, no inflight leak
        assertEquals(0, queueSize(PREFILL_IP_PORT));
        assertEquals(0, decodeEp.getInflightCount());
        assertTrue(scheduler.registerInflight(item(13, 70, 0, PREFILL_IP_PORT)),
                "request id must not stay inflight/tombstoned after a failed rescue");
        verify(priorityReporter).reportRescue(eq(70), eq("requeue_failed"));
        verify(priorityReporter).reportTransfer(eq(70), eq("requeue_failed"));
        // §14.4 latency metric on failure: to_endpoint downgraded to "-"
        verify(priorityReporter).reportRescueLatency(eq(70), eq("requeue_failed"),
                eq(PREFILL_IP_PORT), eq("-"), anyLong());
    }

    // ==================== scanner lifecycle ====================

    @Test
    void scanner_thread_not_created_unless_both_switches_are_on() {
        config.setAutoTpmEnabled(true);
        config.setAutoTpmDeadlineRescueEnabled(false);
        service.start();
        assertFalse(service.isRunning());

        config.setAutoTpmEnabled(false);
        config.setAutoTpmDeadlineRescueEnabled(true);
        service.start();
        assertFalse(service.isRunning());
    }

    @Test
    void scanner_starts_and_stops_cleanly_when_enabled() throws Exception {
        config.setAutoTpmRescueScanIntervalMs(5);
        service.start();
        for (int i = 0; i < 100 && !service.isRunning(); i++) {
            Thread.sleep(10);
        }
        assertTrue(service.isRunning());

        service.stop();
        assertFalse(service.isRunning());
    }

    // ==================== helpers ====================

    private void registerPrefill(String ipPort, String ip) {
        WorkerStatus ws = new WorkerStatus();
        ws.setIp(ip);
        ws.setPort(8080);
        ws.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, ipPort, ws);
    }

    private PrefillEndpoint prefill(String ipPort) {
        return endpointRegistry.getPrefill(ipPort);
    }

    private int queueSize(String ipPort) {
        return prefill(ipPort).getBatcher().queueSize();
    }

    /**
     * Register + reserve + enqueue one request exactly like a committed
     * Auto-TPM plan would leave it: inflight-registered, decode reserved,
     * queued on the prefill batcher with priority/deadline set.
     */
    private BatchItem enqueueDanger(long requestId, int priority, long deadlineMs, String prefillIpPort) {
        BatchItem item = item(requestId, priority, deadlineMs, prefillIpPort);
        assertTrue(scheduler.registerInflight(item));
        endpointRegistry.getDecode(DECODE_IP_PORT).reserve(requestId, 128, 136);
        assertTrue(prefill(prefillIpPort).getBatcher().tryOffer(item));
        return item;
    }

    private BatchItem item(long requestId, int priority, long deadlineMs, String prefillIpPort) {
        String prefillIp = prefillIpPort.split(":")[0];
        Response route = successRoute(requestId, prefillIp);
        BalanceContext ctx = context(requestId, priority);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, ctx.getStartTime(), deadlineMs));
        BatchItem item = new BatchItem(ctx, new CompletableFuture<>(), route,
                FlexlbBatchScheduler.findServer(route, RoleType.PREFILL),
                FlexlbBatchScheduler.findServer(route, RoleType.DECODE),
                prefill(prefillIpPort), endpointRegistry.getDecode(DECODE_IP_PORT),
                System.currentTimeMillis());
        return item;
    }

    private static BalanceContext context(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(new FlexlbConfig());
        return ctx;
    }

    private static Response successRoute(long requestId, String prefillIp) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(
                server(RoleType.PREFILL, prefillIp, 8080, 8081, requestId),
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
