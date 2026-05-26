package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.DpBatchScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * V1-α: verify {@link RouteService} dispatches to the right downstream:
 *
 *  - dpBalanceEnabled=true && pd_sep gate ok → DpBatchScheduler
 *  - dpBalanceEnabled=true && SP/beam       → fall through to queue or direct
 *  - dpBalanceEnabled=false && enableQueueing=true → QueueManager
 *  - dpBalanceEnabled=false && enableQueueing=false → DefaultRouter (sync)
 *
 *  Plus: cancel always cascades to DpBatchScheduler (no-op for unknown ids,
 *        so it's safe under all configs).
 */
class RouteServiceTest {

    private ConfigService configService;
    private FlexlbConfig cfg;
    private DefaultRouter defaultRouter;
    private QueueManager queueManager;
    private DpBatchScheduler dpBatchScheduler;
    private EngineWorkerStatus engineWorkerStatus;
    private RouteService routeService;

    @BeforeEach
    void setUp() {
        configService = mock(ConfigService.class);
        defaultRouter = mock(DefaultRouter.class);
        queueManager = mock(QueueManager.class);
        dpBatchScheduler = mock(DpBatchScheduler.class);
        engineWorkerStatus = mock(EngineWorkerStatus.class);

        cfg = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(cfg);
        when(engineWorkerStatus.selectModelWorkerStatus(any(RoleType.class), any()))
                .thenReturn(workersWithDpSize(4));

        routeService = new RouteService(configService, defaultRouter, queueManager,
                engineWorkerStatus, dpBatchScheduler);
    }

    private static Map<String, WorkerStatus> workersWithDpSize(long dpSize) {
        Map<String, WorkerStatus> m = new HashMap<>();
        WorkerStatus w = new WorkerStatus();
        w.setIp("10.0.0.1");
        w.setPort(8080);
        w.setDpSize(dpSize);
        m.put("10.0.0.1:8080", w);
        return m;
    }

    // ============== route dispatch ==============

    @Test
    void dpBalance_off_falls_through_to_direct_router() {
        cfg.setDpBalanceEnabled(false);
        cfg.setEnableQueueing(false);
        when(defaultRouter.route(any())).thenReturn(okResponse());

        BalanceContext ctx = ctxWith(maxNewTokens(128));
        Response r = routeService.route(ctx).block();

        assertNotNull(r);
        assertTrue(r.isSuccess());
        verify(defaultRouter, times(1)).route(any());
        verify(dpBatchScheduler, never()).submit(any());
        verify(queueManager, never()).tryRouteAsync(any());
    }

    @Test
    void dpBalance_off_queueing_on_routes_to_queue_manager() {
        cfg.setDpBalanceEnabled(false);
        cfg.setEnableQueueing(true);
        when(queueManager.tryRouteAsync(any())).thenReturn(Mono.just(okResponse()));

        BalanceContext ctx = ctxWith(maxNewTokens(128));
        Response r = routeService.route(ctx).block();

        assertTrue(r.isSuccess());
        verify(queueManager).tryRouteAsync(any());
        verify(dpBatchScheduler, never()).submit(any());
        verify(defaultRouter, never()).route(any());
    }

    @Test
    void dpBalance_on_with_normal_request_routes_to_dp_batch_scheduler() {
        cfg.setDpBalanceEnabled(true);
        cfg.setEnableQueueing(true);  // even with queueing on, DP takes priority
        when(dpBatchScheduler.submit(any())).thenReturn(CompletableFuture.completedFuture(okResponse()));

        BalanceContext ctx = ctxWith(maxNewTokens(128));
        Response r = routeService.route(ctx).block();

        assertTrue(r.isSuccess());
        verify(dpBatchScheduler).submit(any());
        verify(queueManager, never()).tryRouteAsync(any());
        verify(defaultRouter, never()).route(any());
    }

    @Test
    void dpBalance_on_but_max_new_tokens_one_falls_through() {
        // max_new_tokens=1 → pd_separation is off → DP batching is not applicable
        cfg.setDpBalanceEnabled(true);
        cfg.setEnableQueueing(false);
        when(defaultRouter.route(any())).thenReturn(okResponse());

        BalanceContext ctx = ctxWith(maxNewTokens(1));
        routeService.route(ctx).block();

        verify(defaultRouter).route(any());
        verify(dpBatchScheduler, never()).submit(any());
    }

    @Test
    void dpBalance_on_but_beam_search_falls_through() {
        cfg.setDpBalanceEnabled(true);
        cfg.setEnableQueueing(false);
        when(defaultRouter.route(any())).thenReturn(okResponse());

        Request req = maxNewTokens(128);
        req.setNumBeams(4);
        BalanceContext ctx = ctxWith(req);
        routeService.route(ctx).block();

        verify(defaultRouter).route(any());
        verify(dpBatchScheduler, never()).submit(any());
    }

    @Test
    void dpBalance_on_but_force_disable_sp_run_falls_through() {
        cfg.setDpBalanceEnabled(true);
        cfg.setEnableQueueing(false);
        when(defaultRouter.route(any())).thenReturn(okResponse());

        Request req = maxNewTokens(128);
        req.setForceDisableSpRun(true);
        BalanceContext ctx = ctxWith(req);
        routeService.route(ctx).block();

        verify(defaultRouter).route(any());
        verify(dpBatchScheduler, never()).submit(any());
    }

    @Test
    void dpBalanceScheduler_null_falls_through_safely() {
        // Defensive: if the bean isn't wired (early-stage build), we never NPE
        RouteService rs = new RouteService(configService, defaultRouter, queueManager,
                engineWorkerStatus, null);
        cfg.setDpBalanceEnabled(true);
        when(defaultRouter.route(any())).thenReturn(okResponse());

        BalanceContext ctx = ctxWith(maxNewTokens(128));
        Response r = rs.route(ctx).block();

        assertTrue(r.isSuccess());
        verify(defaultRouter).route(any());
    }

    @Test
    void dpBalance_on_single_rank_worker_routes_to_dp_batch_scheduler() {
        cfg.setDpBalanceEnabled(true);
        cfg.setEnableQueueing(false);
        when(engineWorkerStatus.selectModelWorkerStatus(any(RoleType.class), any()))
                .thenReturn(workersWithDpSize(1));
        when(dpBatchScheduler.submit(any())).thenReturn(CompletableFuture.completedFuture(okResponse()));

        BalanceContext ctx = ctxWith(maxNewTokens(128));
        Response r = routeService.route(ctx).block();

        assertTrue(r.isSuccess());
        verify(dpBatchScheduler).submit(any());
        verify(defaultRouter, never()).route(any());
    }

    @Test
    void dpBalance_on_but_no_prefill_worker_falls_through_to_legacy() {
        cfg.setDpBalanceEnabled(true);
        cfg.setEnableQueueing(false);
        when(engineWorkerStatus.selectModelWorkerStatus(any(RoleType.class), any()))
                .thenReturn(new HashMap<>());
        when(defaultRouter.route(any())).thenReturn(okResponse());

        BalanceContext ctx = ctxWith(maxNewTokens(128));
        routeService.route(ctx).block();

        verify(defaultRouter).route(any());
        verify(dpBatchScheduler, never()).submit(any());
    }

    // ============== gate function direct ==============

    @Test
    void shouldUseDpBatch_returns_true_only_when_all_gates_pass() {
        cfg.setDpBalanceEnabled(true);
        BalanceContext ctx = ctxWith(maxNewTokens(128));
        assertTrue(routeService.shouldUseDpBatch(ctx, cfg));
    }

    @Test
    void shouldUseDpBatch_returns_false_when_request_null() {
        cfg.setDpBalanceEnabled(true);
        BalanceContext ctx = new BalanceContext();
        assertFalse(routeService.shouldUseDpBatch(ctx, cfg));
    }

    // ============== cancel cascading ==============

    @Test
    void cancel_cascades_to_dp_batch_scheduler_when_request_present() {
        cfg.setEnableQueueing(false);
        BalanceContext ctx = ctxWith(maxNewTokens(128));
        ctx.getRequest().setRequestId(42L);

        routeService.cancel(ctx);

        verify(dpBatchScheduler).cancel(42L);
        assertFalse(ctx.isSuccess());
        assertEquals("request cancelled", ctx.getErrorMessage());
    }

    @Test
    void cancel_with_queueing_completes_future_exceptionally() {
        cfg.setEnableQueueing(true);
        BalanceContext ctx = ctxWith(maxNewTokens(128));
        ctx.getRequest().setRequestId(7L);
        CompletableFuture<Response> f = new CompletableFuture<>();
        ctx.setFuture(f);

        routeService.cancel(ctx);

        assertTrue(f.isCompletedExceptionally());
        assertTrue(ctx.isCancelled(), "BalanceContext.cancelled flag must flip");
        verify(dpBatchScheduler).cancel(7L);
    }

    @Test
    void cancel_with_no_dp_scheduler_does_not_NPE() {
        RouteService rs = new RouteService(configService, defaultRouter, queueManager,
                engineWorkerStatus, null);
        cfg.setEnableQueueing(false);
        BalanceContext ctx = ctxWith(maxNewTokens(128));
        ctx.getRequest().setRequestId(7L);

        rs.cancel(ctx);  // must not throw

        assertFalse(ctx.isSuccess());
    }

    // ============== helpers ==============

    private static Response okResponse() {
        Response r = new Response();
        r.setSuccess(true);
        r.setCode(200);
        return r;
    }

    private static Request maxNewTokens(int n) {
        Request req = new Request();
        req.setRequestId(1);
        req.setMaxNewTokens(n);
        req.setNumBeams(1);
        return req;
    }

    private static BalanceContext ctxWith(Request req) {
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(req);
        return ctx;
    }
}
