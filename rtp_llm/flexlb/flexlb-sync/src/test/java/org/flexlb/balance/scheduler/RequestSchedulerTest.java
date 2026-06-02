package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for RequestScheduler routing logic.
 * Tests handleRoutingResult behavior including retry limits.
 */
@ExtendWith(MockitoExtension.class)
class RequestSchedulerTest {

    @Mock
    private Router router;
    @Mock
    private ConfigService configService;
    @Mock
    private QueueManager queueManager;
    @Mock
    private DynamicWorkerManager dynamicWorkerManager;
    @Mock
    private RoutingQueueReporter metrics;

    private RequestScheduler scheduler;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = new FlexlbConfig();
        config.setScheduleWorkerSize(1);
        config.setMaxRetryCount(3); // Explicitly set for test, default is 0 (unlimited)
        lenient().when(configService.loadBalanceConfig()).thenReturn(config);
        scheduler = new RequestScheduler(router, configService, queueManager, dynamicWorkerManager, metrics);
    }

    @Test
    void processRequest_shouldCompleteOnSuccess() throws Exception {
        BalanceContext ctx = createContext(1L);
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        when(router.route(ctx)).thenReturn(successResponse);

        // Use reflection to invoke private processRequest
        var method = RequestScheduler.class.getDeclaredMethod("processRequest", BalanceContext.class);
        method.setAccessible(true);
        method.invoke(scheduler, ctx);

        assertTrue(ctx.getFuture().isDone());
        assertTrue(ctx.getFuture().get().isSuccess());
        verify(metrics).reportRoutingSuccessQps(0);
    }

    @Test
    void processRequest_shouldRetryOnRetryableError() throws Exception {
        BalanceContext ctx = createContext(1L);
        Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        when(router.route(ctx)).thenReturn(errorResponse);

        var method = RequestScheduler.class.getDeclaredMethod("processRequest", BalanceContext.class);
        method.setAccessible(true);
        method.invoke(scheduler, ctx);

        assertEquals(1, ctx.getRetryCount());
        verify(queueManager).offerToHead(ctx);
        verify(metrics).reportRoutingFailureQps(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
    }

    @Test
    void processRequest_shouldNotRetryOnNonRetryableError() throws Exception {
        BalanceContext ctx = createContext(1L);
        Response errorResponse = Response.error(StrategyErrorType.INVALID_REQUEST);
        when(router.route(ctx)).thenReturn(errorResponse);

        var method = RequestScheduler.class.getDeclaredMethod("processRequest", BalanceContext.class);
        method.setAccessible(true);
        method.invoke(scheduler, ctx);

        assertEquals(0, ctx.getRetryCount());
        verify(queueManager, never()).offerToHead(any());
        assertTrue(ctx.getFuture().isDone());
        assertFalse(ctx.getFuture().get().isSuccess());
    }

    @Test
    void processRequest_shouldStopRetryingAfterMaxRetries() throws Exception {
        BalanceContext ctx = createContext(1L);
        // Simulate already retried 3 times (max)
        for (int i = 0; i < 3; i++) {
            ctx.incrementRetryCount();
        }

        Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        when(router.route(ctx)).thenReturn(errorResponse);

        var method = RequestScheduler.class.getDeclaredMethod("processRequest", BalanceContext.class);
        method.setAccessible(true);
        method.invoke(scheduler, ctx);

        // Should NOT re-queue, should complete with error
        verify(queueManager, never()).offerToHead(any());
        assertTrue(ctx.getFuture().isDone());
        assertFalse(ctx.getFuture().get().isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), ctx.getFuture().get().getCode());
    }

    @Test
    void processRequest_shouldCompleteExceptionallyOnException() throws Exception {
        BalanceContext ctx = createContext(1L);
        when(router.route(ctx)).thenThrow(new RuntimeException("routing error"));

        var method = RequestScheduler.class.getDeclaredMethod("processRequest", BalanceContext.class);
        method.setAccessible(true);
        method.invoke(scheduler, ctx);

        assertTrue(ctx.getFuture().isCompletedExceptionally());
    }

    private BalanceContext createContext(long requestId) {
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setGenerateTimeout(60_000);
        ctx.setRequest(request);
        ctx.setFuture(new CompletableFuture<>());

        FlexlbConfig config = new FlexlbConfig();
        config.setMaxRetryCount(3);
        ctx.setConfig(config);
        return ctx;
    }
}
