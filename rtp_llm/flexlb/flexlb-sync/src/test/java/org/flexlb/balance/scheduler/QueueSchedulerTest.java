package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.FlexlbMetricHelper;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link QueueScheduler}: retry-aware route completion
 * (migrated from the former RequestSchedulerTest, which exercised
 * RequestScheduler.processRequest via reflection — now the inherited
 * {@code routeAndComplete} + overridden {@code onRouteResult}) plus the
 * submit-side reactive pipeline (queue-full fast path, generate-timeout).
 */
@ExtendWith(MockitoExtension.class)
class QueueSchedulerTest {

    @Mock
    private Router router;
    @Mock
    private ConfigService configService;
    @Mock
    private DynamicWorkerManager dynamicWorkerManager;
    @Mock
    private RoutingQueueReporter metrics;

    private FlexlbConfig config;
    private InflightStore inflightStore;
    private QueueScheduler scheduler;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setMaxQueueSize(10);
        config.setMaxRetryCount(3); // Explicitly set for test, default is 0 (unlimited)
        lenient().when(configService.loadBalanceConfig()).thenReturn(config);
        inflightStore = new InflightStore(mock(BatchSchedulerReporter.class), configService);
        scheduler = new QueueScheduler(router, configService, metrics, dynamicWorkerManager,
                inflightStore, new FlexlbMetricHelper(null, MetricConstant.PATH_QUEUE));
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
        inflightStore.shutdown();
    }

    // ==================== Route completion policy (worker consume path) ====================

    @Test
    void routeAndComplete_shouldCompleteOnSuccess() throws Exception {
        BalanceContext ctx = createContext(1L);
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        when(router.route(ctx)).thenReturn(successResponse);

        scheduler.routeAndComplete(ctx, ctx.getFuture());

        assertTrue(ctx.getFuture().isDone());
        assertTrue(ctx.getFuture().get().isSuccess());
        verify(metrics).reportRoutingSuccessQps(0);
    }

    @Test
    void routeAndComplete_shouldRetryOnRetryableError() {
        BalanceContext ctx = createContext(1L);
        Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        when(router.route(ctx)).thenReturn(errorResponse);

        scheduler.routeAndComplete(ctx, ctx.getFuture());

        assertEquals(1, ctx.getRetryCount());
        assertFalse(ctx.getFuture().isDone());
        assertEquals(1, scheduler.queueSize()); // re-queued at the head
        verify(metrics).reportRoutingFailureQps(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
    }

    @Test
    void routeAndComplete_shouldNotRetryOnNonRetryableError() throws Exception {
        BalanceContext ctx = createContext(1L);
        Response errorResponse = Response.error(StrategyErrorType.INVALID_REQUEST);
        when(router.route(ctx)).thenReturn(errorResponse);

        scheduler.routeAndComplete(ctx, ctx.getFuture());

        assertEquals(0, ctx.getRetryCount());
        assertEquals(0, scheduler.queueSize());
        assertTrue(ctx.getFuture().isDone());
        assertFalse(ctx.getFuture().get().isSuccess());
    }

    @Test
    void routeAndComplete_shouldStopRetryingAfterMaxRetries() throws Exception {
        BalanceContext ctx = createContext(1L);
        // Simulate already retried 3 times (max)
        for (int i = 0; i < 3; i++) {
            ctx.incrementRetryCount();
        }

        Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        when(router.route(ctx)).thenReturn(errorResponse);

        scheduler.routeAndComplete(ctx, ctx.getFuture());

        // Should NOT re-queue, should complete with error
        assertEquals(0, scheduler.queueSize());
        assertTrue(ctx.getFuture().isDone());
        assertFalse(ctx.getFuture().get().isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), ctx.getFuture().get().getCode());
    }

    @Test
    void routeAndComplete_shouldCompleteExceptionallyOnException() {
        BalanceContext ctx = createContext(1L);
        when(router.route(ctx)).thenThrow(new RuntimeException("routing error"));

        scheduler.routeAndComplete(ctx, ctx.getFuture());

        assertTrue(ctx.getFuture().isCompletedExceptionally());
    }

    // ==================== Submit-side reactive pipeline ====================

    @Test
    void submit_shouldReturnQueueFullWhenQueueIsFull() throws Exception {
        config.setMaxQueueSize(1);
        // Rebuild the scheduler so the bounded deque picks up the new capacity
        scheduler = new QueueScheduler(router, configService, metrics, dynamicWorkerManager,
                inflightStore, new FlexlbMetricHelper(null, MetricConstant.PATH_QUEUE));

        // Workers not started — the first request stays queued
        CompletableFuture<Response> first = scheduler.submit(createContext(1L));
        assertFalse(first.isDone());

        Response rejected = scheduler.submit(createContext(2L)).get(1, TimeUnit.SECONDS);
        assertFalse(rejected.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), rejected.getCode());
        verify(metrics).reportRejected();
    }

    @Test
    void submit_shouldTimeoutQueuedRequestAfterGenerateTimeout() throws Exception {
        // Workers not started — the request waits in the queue until the
        // reactive pipeline's generate-timeout fires.
        BalanceContext ctx = createContext(1L);
        ctx.getRequest().setGenerateTimeout(100);

        Response response = scheduler.submit(ctx).get(3, TimeUnit.SECONDS);

        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_TIMEOUT.getErrorCode(), response.getCode());
        assertEquals(0, scheduler.queueSize()); // removed from the queue on timeout
        verify(metrics).reportTimeout();
    }

    // ==================== Cancel cascade (queue slot release) ====================

    @Test
    void cancel_shouldReleaseQueueSlotThroughOnCancelCascade() {
        // Workers not started — the request stays queued, occupying a slot.
        CompletableFuture<Response> result = scheduler.submit(createContext(1L));
        assertEquals(1, scheduler.queueSize());

        assertTrue(scheduler.cancel("1"));

        assertEquals(0, scheduler.queueSize()); // slot freed, not dead-occupied
        assertTrue(result.isCompletedExceptionally());
        assertTrue(inflightStore.get("1").isTerminated());
    }

    @Test
    void cancel_onCancelCascadeIsBestEffortIdempotent() {
        scheduler.submit(createContext(1L));
        InflightItem item = inflightStore.get("1");

        assertTrue(item.cancel());
        item.scheduler().onCancel(item); // queued → removed
        assertEquals(0, scheduler.queueSize());
        // A second cascade against an absent entry must not throw.
        item.scheduler().onCancel(item);
        assertEquals(0, scheduler.queueSize());
    }

    private BalanceContext createContext(long requestId) {
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setGenerateTimeout(60_000);
        ctx.setRequest(request);
        ctx.setFuture(new CompletableFuture<>());
        ctx.setConfig(config);
        return ctx;
    }
}
