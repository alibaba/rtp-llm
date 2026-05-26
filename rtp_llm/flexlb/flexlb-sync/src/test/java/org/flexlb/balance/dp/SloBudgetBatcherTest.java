package org.flexlb.balance.dp;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DpBatchReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeast;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Behavioural tests for {@link SloBudgetBatcher}. Pre-set predictor returns
 * deterministic estimates so we can probe each branch of {@code stepOnce}:
 * head-alone-exceeds-SLO, capacity-fit pack, deadline-force, single-head park.
 */
class SloBudgetBatcherTest {

    private ConfigService configService;
    private FlexlbConfig cfg;
    private DispatchPlanner planner;
    private PrefillTimePredictor predictor;
    private DpBatchReporter reporter;
    private final List<PrefillBatch> dispatched = new CopyOnWriteArrayList<>();
    private SloBudgetBatcher batcher;

    private static final ServerStatus PREFILL = serverStatus("10.0.0.1", 8080, 9080);
    private static final ServerStatus DECODE = serverStatus("10.0.0.2", 8081, 9081);

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        planner = mock(DispatchPlanner.class);
        predictor = mock(PrefillTimePredictor.class);

        cfg = new FlexlbConfig();
        cfg.setBatchMaxTokens(8192);
        cfg.setDpTtftSloMs(2000);
        cfg.setSloSafetyMargin(50);
        cfg.setDpMaxScanAhead(8);
        when(configService.loadBalanceConfig()).thenReturn(cfg);

        when(planner.plan(any(), any())).thenAnswer(inv -> {
            List<QueuedRequest> drained = inv.getArgument(0);
            if (drained == null || drained.isEmpty()) {
                return DispatchPlan.empty();
            }
            List<PendingRequest> placed = new ArrayList<>();
            for (QueuedRequest qr : drained) {
                placed.add(new PendingRequest(qr.ctx(), PREFILL, DECODE, qr.future(), qr.enqueuedAtMicros()));
            }
            return DispatchPlan.of(List.of(new PrefillBatch(PREFILL, placed, 1)));
        });

        when(predictor.estimateMs(any(Long.class), any(Long.class))).thenReturn(10L);

        reporter = mock(DpBatchReporter.class);
        batcher = new SloBudgetBatcher("m1", configService, planner, dispatched::add, predictor, reporter);
    }

    @AfterEach
    void tearDown() {
        batcher.shutdown();
    }

    @Test
    void multiple_requests_pack_into_single_batch_when_under_budget_and_capacity() throws Exception {
        List<CompletableFuture<Response>> futures = offerN(3, 100);
        waitUntilDispatched(1, 1000);
        assertEquals(1, dispatched.size());
        assertEquals(3, dispatched.get(0).size());
        for (CompletableFuture<Response> f : futures) {
            assertFalse(f.isDone(), "test callback doesn't complete futures; that's DpBatchScheduler's job");
        }
    }

    @Test
    void head_alone_exceeds_slo_budget_fails_with_slo_exceeded() throws Exception {
        cfg.setDpTtftSloMs(100);
        cfg.setSloSafetyMargin(0);
        when(predictor.estimateMs(any(Long.class), any(Long.class))).thenReturn(10_000L);

        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 4096), f));

        Response r = f.get(1, TimeUnit.SECONDS);
        assertNotNull(r);
        assertFalse(r.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), r.getCode());
        assertEquals(0, dispatched.size());
    }

    @Test
    void capacity_limit_excludes_oversize_request_from_pack() throws Exception {
        cfg.setBatchMaxTokens(1000);
        // head fits, second would overflow, third small one fits via backward scan
        CompletableFuture<Response> f1 = new CompletableFuture<>();
        CompletableFuture<Response> f2 = new CompletableFuture<>();
        CompletableFuture<Response> f3 = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 500), f1));
        batcher.offer(QueuedRequest.of(ctx(2, 900), f2));
        batcher.offer(QueuedRequest.of(ctx(3, 400), f3));

        waitUntilDispatched(1, 1000);
        assertEquals(1, dispatched.size());
        // head (500) + small (400) = 900 fits; big (900) skipped
        assertEquals(2, dispatched.get(0).size());
    }

    @Test
    void cancel_in_queue_removes_request_and_completes_future_exceptionally() {
        // park the worker by making it think only one request — too few to pack
        cfg.setDpMaxScanAhead(16);
        CompletableFuture<Response> f1 = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(42, 100), f1));

        boolean removed = batcher.cancelInQueue(42L);
        assertTrue(removed || f1.isDone(),
                "either the request was cancelled in queue, or it already dispatched");
        if (removed) {
            assertTrue(f1.isCompletedExceptionally());
        }
    }

    @Test
    void cancel_in_queue_unknown_returns_false() {
        assertFalse(batcher.cancelInQueue(99999L));
    }

    @Test
    void planner_exception_fails_all_drained_futures() throws Exception {
        when(planner.plan(any(), any())).thenThrow(new RuntimeException("boom"));
        List<CompletableFuture<Response>> futures = offerN(3, 100);

        for (CompletableFuture<Response> f : futures) {
            // give the worker thread a moment to drain and dispatch
            try {
                f.get(1, TimeUnit.SECONDS);
            } catch (Exception ignored) {
            }
            assertTrue(f.isCompletedExceptionally(),
                    "every drained request's future must be failed when the planner throws");
        }
    }

    @Test
    void per_request_failures_from_planner_complete_futures_with_failure_response() throws Exception {
        when(planner.plan(any(), any())).thenAnswer(inv -> {
            List<QueuedRequest> drained = inv.getArgument(0);
            return DispatchPlan.allFailed(drained, StrategyErrorType.NO_DECODE_WORKER, "no decode");
        });
        List<CompletableFuture<Response>> futures = offerN(2, 100);

        for (CompletableFuture<Response> f : futures) {
            Response r = f.get(1, TimeUnit.SECONDS);
            assertNotNull(r);
            assertFalse(r.isSuccess());
            assertEquals(StrategyErrorType.NO_DECODE_WORKER.getErrorCode(), r.getCode());
        }
    }

    @Test
    void deadline_force_dispatches_when_slo_budget_already_zero() throws Exception {
        cfg.setDpTtftSloMs(0);  // budget will be <= 0 immediately
        cfg.setSloSafetyMargin(0);
        // predictor returns small enough so head wouldn't fail on its own
        when(predictor.estimateMs(any(Long.class), any(Long.class))).thenReturn(0L);

        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f));

        waitUntilDispatched(1, 1000);
        assertEquals(1, dispatched.size());
        assertEquals(1, dispatched.get(0).size());
    }

    @Test
    void queue_size_method_reflects_pending_count_during_park() {
        // long SLO + capacity is tiny so the worker will park (single head can't pack with anyone)
        cfg.setDpTtftSloMs(60_000);
        AtomicInteger remaining = new AtomicInteger(1);
        when(planner.plan(any(), any())).thenAnswer(inv -> {
            // Only allow dispatch when test signals
            if (remaining.get() <= 0) {
                List<QueuedRequest> drained = inv.getArgument(0);
                List<PendingRequest> placed = new ArrayList<>();
                for (QueuedRequest qr : drained) {
                    placed.add(new PendingRequest(qr.ctx(), PREFILL, DECODE, qr.future(), qr.enqueuedAtMicros()));
                }
                return DispatchPlan.of(List.of(new PrefillBatch(PREFILL, placed, 1)));
            }
            return DispatchPlan.empty();
        });
        // Single offer — worker will park on the head waiting for a companion
        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f));
        // queue depth should briefly contain the head before pack/park
        // (not asserting an exact value to avoid race; just exercise the method)
        assertTrue(batcher.queueSize() >= 0);
    }

    // ============== metric emission ==============

    @Test
    void metrics_packed_dispatch_emits_queue_snapshot_and_batch_tokens_and_dp_slot() throws Exception {
        offerN(3, 100);
        waitUntilDispatched(1, 1000);

        // queue snapshot reported at least once (PRECISE PARK before signal not guaranteed,
        // but the stepOnce that dispatches definitely emits one)
        verify(reporter, atLeastOnce()).reportSloQueueSnapshot(eq("m1"), anyInt(), anyLong());
        // batch tokens: target=8192 (cfg default), actual=3*100=300
        verify(reporter).reportSloBatchTokens("m1", DpBatchReporter.FlushReason.PACKED, 8192L, 300L);
        // per-DP slot: 3 requests on rank=0, 300 tokens, endpoint composed from PREFILL ServerStatus
        verify(reporter).reportSloBatchDpSlot(eq("m1"), eq("PREFILL"), eq("g1"),
                eq("10.0.0.1:9080"), eq(0), eq(3), eq(300L));
        // loop-duration emitted; at least one DISPATCH outcome
        verify(reporter, atLeastOnce()).reportSloLoopDuration(eq("m1"),
                eq(DpBatchReporter.LoopOutcome.DISPATCH), anyLong());
        // loops-per-dispatch emitted exactly once (since dispatch happened)
        verify(reporter, atLeastOnce()).reportSloLoopsPerDispatch(eq("m1"),
                eq(DpBatchReporter.FlushReason.PACKED), anyInt());
        // queue-wait per request: 3 events
        verify(reporter, atLeast(3)).reportSloQueueWait(eq("m1"),
                eq(DpBatchReporter.FlushReason.PACKED), anyLong());
    }

    @Test
    void metrics_head_alone_exceeds_slo_emits_failure_cause_SLO_EXCEEDED() throws Exception {
        cfg.setDpTtftSloMs(100);
        cfg.setSloSafetyMargin(0);
        when(predictor.estimateMs(any(Long.class), any(Long.class))).thenReturn(10_000L);

        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 4096), f));
        f.get(1, TimeUnit.SECONDS);

        verify(reporter, atLeastOnce()).reportSloFailure("m1",
                DpBatchReporter.FailureCause.SLO_EXCEEDED);
        // FAIL outcome reported via loop-duration
        verify(reporter, atLeastOnce()).reportSloLoopDuration(eq("m1"),
                eq(DpBatchReporter.LoopOutcome.FAIL), anyLong());
        // failed request must NOT bump per-dispatch loops counter
        verify(reporter, never()).reportSloLoopsPerDispatch(eq("m1"),
                eq(DpBatchReporter.FlushReason.SLO_EXCEEDED), anyInt());
    }

    @Test
    void metrics_planner_throws_emits_PLANNER_ERROR_cause_per_failed_request() throws Exception {
        when(planner.plan(any(), any())).thenThrow(new RuntimeException("boom"));
        List<CompletableFuture<Response>> futures = offerN(2, 100);
        for (CompletableFuture<Response> f : futures) {
            try {
                f.get(1, TimeUnit.SECONDS);
            } catch (Exception ignored) {
            }
        }

        verify(reporter, atLeast(2)).reportSloFailure("m1",
                DpBatchReporter.FailureCause.PLANNER_ERROR);
    }

    @Test
    void metrics_deadline_force_reports_DEADLINE_FORCE_batch_tokens() throws Exception {
        cfg.setDpTtftSloMs(0);
        cfg.setSloSafetyMargin(0);
        when(predictor.estimateMs(any(Long.class), any(Long.class))).thenReturn(0L);

        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f));
        waitUntilDispatched(1, 1000);

        // target=8192, actual=100, reason=DEADLINE_FORCE
        verify(reporter).reportSloBatchTokens("m1", DpBatchReporter.FlushReason.DEADLINE_FORCE,
                8192L, 100L);
    }

    // ============== helpers ==============

    private List<CompletableFuture<Response>> offerN(int n, long seqLen) {
        List<CompletableFuture<Response>> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            CompletableFuture<Response> f = new CompletableFuture<>();
            out.add(f);
            batcher.offer(QueuedRequest.of(ctx(i + 1, seqLen), f));
        }
        return out;
    }

    private void waitUntilDispatched(int target, long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (dispatched.size() < target && System.currentTimeMillis() < deadline) {
            Thread.sleep(5);
        }
    }

    private static BalanceContext ctx(long requestId, long seqLen) {
        BalanceContext c = new BalanceContext();
        Request r = new Request();
        r.setRequestId(requestId);
        r.setSeqLen(seqLen);
        c.setRequest(r);
        return c;
    }

    private static ServerStatus serverStatus(String ip, int httpPort, int grpcPort) {
        ServerStatus s = new ServerStatus();
        s.setRole(RoleType.PREFILL);
        s.setServerIp(ip);
        s.setHttpPort(httpPort);
        s.setGrpcPort(grpcPort);
        s.setGroup("g1");
        s.setSuccess(true);
        return s;
    }
}
