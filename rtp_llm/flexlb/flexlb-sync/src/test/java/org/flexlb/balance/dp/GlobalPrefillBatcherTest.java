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
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class GlobalPrefillBatcherTest {

    private ScheduledExecutorService timer;
    private ConfigService configService;
    private FlexlbConfig cfg;
    private EngineWorkerStatus engineWorkerStatus;
    private DispatchPlanner planner;
    private final List<PrefillBatch> dispatched = new CopyOnWriteArrayList<>();
    private GlobalPrefillBatcher batcher;

    private static final ServerStatus PREFILL = serverStatus("10.0.0.1", 8080, 9080);
    private static final ServerStatus DECODE  = serverStatus("10.0.0.2", 8081, 9081);

    @BeforeEach
    void setUp() {
        timer = Executors.newScheduledThreadPool(1, r -> {
            Thread t = new Thread(r, "test-batcher-timer");
            t.setDaemon(true);
            return t;
        });
        configService = mock(ConfigService.class);
        engineWorkerStatus = mock(EngineWorkerStatus.class);
        planner = mock(DispatchPlanner.class);

        cfg = new FlexlbConfig();
        cfg.setDpBatchSizeMax(4);
        cfg.setDpBatchWindowMs(50);
        cfg.setDpBatchTimeoutMs(500);
        when(configService.loadBalanceConfig()).thenReturn(cfg);

        // Default: planner turns the drain into one PrefillBatch.
        when(planner.plan(any(), any())).thenAnswer(inv -> {
            List<QueuedRequest> drained = inv.getArgument(0);
            DispatchContext ctx = inv.getArgument(1);
            List<PendingRequest> placed = new ArrayList<>();
            for (QueuedRequest qr : drained) {
                placed.add(new PendingRequest(qr.ctx(), PREFILL, DECODE, qr.future(), qr.enqueuedAtMicros()));
            }
            return DispatchPlan.of(List.of(new PrefillBatch(PREFILL, placed, ctx.dpSize())));
        });

        batcher = new GlobalPrefillBatcher("m1", configService, engineWorkerStatus,
                planner, dispatched::add, timer);
    }

    @AfterEach
    void tearDown() {
        timer.shutdownNow();
    }

    @Test
    void size_trigger_flushes_immediately() {
        List<CompletableFuture<Response>> futures = offerN(4);
        // No timer needed — size trigger flushes synchronously on the offer thread.
        assertEquals(1, dispatched.size());
        assertEquals(4, dispatched.get(0).size());
        // Caller still completes futures via the dispatch callback (the test's
        // callback just collects into `dispatched`, it doesn't complete futures).
        for (CompletableFuture<Response> f : futures) {
            assertFalse(f.isDone(), "test callback doesn't complete futures; that's DpBatchScheduler's job");
        }
    }

    @Test
    void window_timer_flushes_partial_batch() throws Exception {
        cfg.setDpBatchWindowMs(20);
        offerN(2);

        // Partial batch — wait past the window for the timer to fire.
        Thread.sleep(80);

        assertEquals(1, dispatched.size());
        assertEquals(2, dispatched.get(0).size());
    }

    @Test
    void per_request_timeout_force_flushes_starving_head() throws Exception {
        cfg.setDpBatchSizeMax(8);  // size threshold not reachable from 1 request
        cfg.setDpBatchWindowMs(60_000);  // window won't fire in this test
        cfg.setDpBatchTimeoutMs(30);

        offerN(1);
        // Wait past the per-request timeout; the next offer triggers force-flush
        // of the starved head.
        Thread.sleep(60);
        offerN(1);

        // The starving head's offer arrival path detected the deadline and drained
        // both queued requests. Either both go in one drain, or the head goes alone
        // and the second triggers another check; the contract is just "starvation
        // is bounded by dpBatchTimeoutMs".
        assertTrue(dispatched.size() >= 1);
        int total = dispatched.stream().mapToInt(PrefillBatch::size).sum();
        assertEquals(2, total);
    }

    @Test
    void cancelInQueue_removes_pending_request_and_fails_future() {
        cfg.setDpBatchSizeMax(8);
        cfg.setDpBatchWindowMs(60_000);
        List<CompletableFuture<Response>> futures = offerN(3);

        long requestId = 2L;
        boolean removed = batcher.cancelInQueue(requestId);

        assertTrue(removed);
        assertEquals(2, batcher.queueSize());
        // Critical: future must be completed so the upstream Mono doesn't hang.
        // Request 2 (indexed by 1-based requestId) is futures.get(1).
        assertTrue(futures.get(1).isCompletedExceptionally(),
                "cancelInQueue must complete the future of the removed request");
        // Other futures (still queued) remain unaffected.
        assertFalse(futures.get(0).isDone());
        assertFalse(futures.get(2).isDone());
    }

    @Test
    void cancelInQueue_unknown_returns_false() {
        offerN(1);
        assertFalse(batcher.cancelInQueue(99999L));
    }

    @Test
    void planner_exception_fails_all_drained_futures() throws Exception {
        // doThrow().when() avoids re-invoking the previously stubbed answer during stubbing.
        doThrow(new RuntimeException("boom")).when(planner).plan(any(), any());
        List<CompletableFuture<Response>> futures = offerN(4);

        for (CompletableFuture<Response> f : futures) {
            assertTrue(f.isCompletedExceptionally(),
                    "every drained request's future must be failed when the planner throws");
        }
    }

    @Test
    void per_request_failures_from_planner_complete_futures_with_failure_response() throws Exception {
        doAnswer(inv -> {
            List<QueuedRequest> drained = inv.getArgument(0);
            return DispatchPlan.allFailed(drained, StrategyErrorType.NO_PREFILL_WORKER, "test");
        }).when(planner).plan(any(), any());
        List<CompletableFuture<Response>> futures = offerN(2);

        for (CompletableFuture<Response> f : futures) {
            Response r = f.get(1, TimeUnit.SECONDS);
            assertNotNull(r);
            assertFalse(r.isSuccess());
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), r.getCode());
        }
    }

    @Test
    void dispatch_callback_exception_fails_only_that_batch() throws Exception {
        AtomicReference<Throwable> dispatchEx = new AtomicReference<>(new RuntimeException("dispatch failed"));
        GlobalPrefillBatcher b = new GlobalPrefillBatcher("m1", configService, engineWorkerStatus,
                planner, batch -> { throw new RuntimeException(dispatchEx.get()); }, timer);

        List<CompletableFuture<Response>> futures = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            CompletableFuture<Response> f = new CompletableFuture<>();
            futures.add(f);
            BalanceContext ctx = ctx(i + 1);
            b.offer(QueuedRequest.of(ctx, f));
        }
        for (CompletableFuture<Response> f : futures) {
            assertTrue(f.isCompletedExceptionally());
        }
    }

    @Test
    void dpSize_falls_back_to_engine_status_when_config_zero() {
        cfg.setDpBatchSizeMax(0);  // auto-detect
        WorkerStatus w = new WorkerStatus();
        w.setIp("10.0.0.1");
        w.setPort(8080);
        w.setDpSize(2);
        w.setAlive(true);
        when(engineWorkerStatus.selectModelWorkerStatus(eq(RoleType.PREFILL), any()))
                .thenReturn(Map.of("10.0.0.1:8080", w));

        // Two offers should size-flush immediately (auto dpSize=2).
        offerN(2);
        assertEquals(1, dispatched.size());
        assertEquals(2, dispatched.get(0).size());
    }

    // ============== helpers ==============

    private List<CompletableFuture<Response>> offerN(int n) {
        List<CompletableFuture<Response>> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            CompletableFuture<Response> f = new CompletableFuture<>();
            out.add(f);
            batcher.offer(QueuedRequest.of(ctx(i + 1), f));
        }
        return out;
    }

    private static BalanceContext ctx(long requestId) {
        BalanceContext c = new BalanceContext();
        Request r = new Request();
        r.setRequestId(requestId);
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
