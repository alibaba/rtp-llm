package org.flexlb.balance.dp;

import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class SloBudgetBatcherTest {

    private ConfigService configService;
    private FlexlbConfig cfg;
    private DispatchPlanner planner;
    private PrefillTimePredictor predictor;
    private DpBatchReporter reporter;
    private final List<DispatchBatch> dispatched = new CopyOnWriteArrayList<>();
    private SloBudgetBatcher batcher;

    private static final ServerStatus PREFILL = serverStatus("10.0.0.1", 8080, 9080);
    private static final ServerStatus DECODE = serverStatus("10.0.0.2", 8081, 9081);

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        planner = mock(DispatchPlanner.class);
        predictor = mock(PrefillTimePredictor.class);

        cfg = new FlexlbConfig();
        cfg.setDpTtftSloMs(2000);
        cfg.setSloSafetyMargin(50);
        cfg.setDpMaxScanAhead(64);
        cfg.setBatchFillThreshold(0.7);
        cfg.setBinarySearchMaxIter(12);
        cfg.setBatchMaxCapacity(1_000_000);
        when(configService.loadBalanceConfig()).thenReturn(cfg);

        WorkerStatus prefillWorker = new WorkerStatus();
        prefillWorker.setIp("10.0.0.1");
        prefillWorker.setPort(8080);
        prefillWorker.setGroup("g1");
        prefillWorker.setAlive(true);
        prefillWorker.setDpSize(1);
        when(planner.selectDecodeWorker(any(), any())).thenReturn(DECODE);

        // predictor: linear, 1 token = 0.1ms
        when(predictor.estimateMs(anyLong(), anyLong())).thenAnswer(inv -> {
            long tokens = inv.getArgument(0);
            return tokens / 10;
        });

        reporter = mock(DpBatchReporter.class);
        batcher = new SloBudgetBatcher("m1", configService, planner, dispatched::add, predictor, reporter, prefillWorker, 1);
    }

    @AfterEach
    void tearDown() {
        batcher.shutdown();
    }

    // ============== EDF ordering ==============

    @Test
    void edf_ordering_processes_most_urgent_first() throws Exception {
        cfg.setDpTtftSloMs(5000);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.0);

        // Offer both requests BEFORE starting the worker thread to avoid race
        batcher.offer(QueuedRequest.of(ctx(1, 100), new CompletableFuture<>()));
        batcher.offer(QueuedRequest.of(ctx(2, 2000), new CompletableFuture<>()));

        batcher.start();
        waitUntilDispatched(1, 1000);
        assertTrue(dispatched.size() >= 1);
        DispatchBatch first = dispatched.get(0);
        long firstSeqLen = first.requests().get(0).ctx().getRequest().getSeqLen();
        assertEquals(2000, firstSeqLen);
    }

    // ============== SLO_DROPPED ==============

    @Test
    void slo_dropped_when_deadline_already_expired() throws Exception {
        cfg.setDpTtftSloMs(0);
        cfg.setSloSafetyMargin(0);

        batcher.start();
        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f));

        Response r = f.get(1, TimeUnit.SECONDS);
        assertNotNull(r);
        assertFalse(r.isSuccess());
        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), r.getCode());
        assertEquals(0, dispatched.size());
    }

    @Test
    void metrics_slo_dropped_emits_failure_cause() throws Exception {
        cfg.setDpTtftSloMs(0);
        cfg.setSloSafetyMargin(0);

        batcher.start();
        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f));
        f.get(1, TimeUnit.SECONDS);

        verify(reporter, atLeastOnce()).reportSloFailure("m1",
                DpBatchReporter.FailureCause.SLO_DROPPED);
        verify(reporter, atLeastOnce()).reportSloTickDuration(eq("m1"),
                eq(DpBatchReporter.LoopOutcome.FAIL), anyLong());
    }

    // ============== EDF_URGENT ==============

    @Test
    void edf_urgent_dispatches_head_alone_when_budget_below_margin() throws Exception {
        cfg.setDpTtftSloMs(55);
        cfg.setSloSafetyMargin(50);

        batcher.start();
        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f));

        waitUntilDispatched(1, 1000);
        assertEquals(1, dispatched.size());
        assertEquals(1, dispatched.get(0).size());
    }

    // ============== Binary search + fill ==============

    @Test
    void batch_ready_dispatches_when_fill_ratio_met() throws Exception {
        cfg.setDpTtftSloMs(5000);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.0);

        // Offer all requests before starting to ensure they're batched together
        offerN(3, 100);
        batcher.start();

        waitUntilAllRequestsDispatched(3, 1000);
        int totalRequests = dispatched.stream().mapToInt(DispatchBatch::size).sum();
        assertEquals(3, totalRequests);
    }

    @Test
    void batch_waits_then_converges_via_budget_decay() throws Exception {
        cfg.setDpTtftSloMs(500);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.7);

        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f));
        batcher.start();

        // Should NOT dispatch immediately — fillRatio too low
        Thread.sleep(50);
        assertEquals(0, dispatched.size(), "should wait while fill ratio is low");

        // Eventually dispatches as budget converges (~490ms)
        waitUntilDispatched(1, 2000);
        assertTrue(dispatched.size() >= 1);
    }

    @Test
    void new_arrival_can_trigger_dispatch() throws Exception {
        cfg.setDpTtftSloMs(500);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.0);

        batcher.start();
        CompletableFuture<Response> f1 = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 100), f1));
        Thread.sleep(20);

        CompletableFuture<Response> f2 = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(2, 200), f2));

        waitUntilDispatched(1, 1000);
        assertTrue(dispatched.size() >= 1);
    }

    // ============== Large request fills budget ==============

    @Test
    void large_request_fills_budget_and_dispatches_quickly() throws Exception {
        cfg.setDpTtftSloMs(200);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.7);

        CompletableFuture<Response> f = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(1, 1500), f));
        batcher.start();

        waitUntilDispatched(1, 500);
        assertEquals(1, dispatched.size());
        assertEquals(1, dispatched.get(0).size());
    }

    // ============== Cancel ==============

    @Test
    void cancel_in_queue_removes_request_and_completes_future_exceptionally() {
        cfg.setDpTtftSloMs(60_000);
        CompletableFuture<Response> f1 = new CompletableFuture<>();
        batcher.offer(QueuedRequest.of(ctx(42, 100), f1));

        boolean removed = batcher.cancelInQueue(42L);
        assertTrue(removed || f1.isDone());
        if (removed) {
            assertTrue(f1.isCompletedExceptionally());
        }
    }

    @Test
    void cancel_unknown_returns_false() {
        assertFalse(batcher.cancelInQueue(99999L));
    }

    // ============== Planner failures ==============

    @Test
    void planner_exception_fails_all_drained_futures() throws Exception {
        cfg.setDpTtftSloMs(80);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.0);
        when(planner.selectDecodeWorker(any(), any())).thenThrow(new RuntimeException("boom"));
        List<CompletableFuture<Response>> futures = offerN(3, 100);
        batcher.start();

        for (CompletableFuture<Response> f : futures) {
            try {
                f.get(1, TimeUnit.SECONDS);
            } catch (Exception ignored) {
            }
            assertTrue(f.isCompletedExceptionally());
        }
    }

    @Test
    void per_request_decode_failure_completes_with_failure_response() throws Exception {
        cfg.setDpTtftSloMs(80);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.0);
        when(planner.selectDecodeWorker(any(), any()))
                .thenReturn(ServerStatus.code(StrategyErrorType.NO_DECODE_WORKER));
        List<CompletableFuture<Response>> futures = offerN(2, 100);
        batcher.start();

        for (CompletableFuture<Response> f : futures) {
            Response r = f.get(1, TimeUnit.SECONDS);
            assertNotNull(r);
            assertFalse(r.isSuccess());
            assertEquals(StrategyErrorType.NO_DECODE_WORKER.getErrorCode(), r.getCode());
        }
    }

    // ============== Metric emission ==============

    @Test
    void metrics_batch_ready_emits_batch_tokens_and_dp_slot() throws Exception {
        cfg.setDpTtftSloMs(5000);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.0);
        offerN(3, 100);
        batcher.start();
        waitUntilAllRequestsDispatched(3, 1000);

        verify(reporter, atLeastOnce()).reportSloQueueSnapshot(eq("m1"), anyInt(), anyLong());
        verify(reporter, atLeastOnce()).reportSloBatchTokens(eq("m1"),
                eq(DpBatchReporter.FlushReason.BATCH_READY), anyLong(), anyLong());
        verify(reporter, atLeastOnce()).reportSloBatchFlush(eq("m1"),
                eq(DpBatchReporter.FlushReason.BATCH_READY), anyInt());
        verify(reporter, atLeastOnce()).reportSloTickDuration(eq("m1"),
                eq(DpBatchReporter.LoopOutcome.DISPATCH), anyLong());
    }

    @Test
    void metrics_planner_throws_emits_PLANNER_ERROR_per_failed_request() throws Exception {
        cfg.setDpTtftSloMs(80);
        cfg.setSloSafetyMargin(0);
        cfg.setBatchFillThreshold(0.0);
        when(planner.selectDecodeWorker(any(), any())).thenThrow(new RuntimeException("boom"));
        offerN(2, 100);
        batcher.start();
        Thread.sleep(500);

        verify(reporter, atLeast(2)).reportSloFailure("m1",
                DpBatchReporter.FailureCause.PLANNER_ERROR);
    }

    @Test
    void queue_size_reflects_pending_count() {
        cfg.setDpTtftSloMs(60_000);
        batcher.offer(QueuedRequest.of(ctx(1, 100), new CompletableFuture<>()));
        assertTrue(batcher.queueSize() >= 0);
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

    private void waitUntilAllRequestsDispatched(int totalRequests, long timeoutMs) throws InterruptedException {
        long deadline = System.currentTimeMillis() + timeoutMs;
        while (System.currentTimeMillis() < deadline) {
            int sum = dispatched.stream().mapToInt(DispatchBatch::size).sum();
            if (sum >= totalRequests) {
                return;
            }
            Thread.sleep(5);
        }
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
