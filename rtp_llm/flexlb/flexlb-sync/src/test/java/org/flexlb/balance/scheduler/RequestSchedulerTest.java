package org.flexlb.balance.scheduler;

import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;

import java.time.Duration;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Unit tests for RequestScheduler routing logic.
 * Tests request completion and retry-limit behavior.
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
        config.setRoutingRetryIntervalMs(0);
        lenient().when(configService.loadBalanceConfig()).thenReturn(config);
        scheduler = new RequestScheduler(router, configService, queueManager, dynamicWorkerManager, metrics);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void processRequest_shouldCompleteOnSuccess() throws Exception {
        BalanceContext ctx = createContext("request-1");
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        when(router.route(ctx)).thenReturn(Mono.just(successResponse));

        scheduler.processRequest(ctx).block();

        assertTrue(ctx.getFuture().isDone());
        assertTrue(ctx.getFuture().get().isSuccess());
        verify(metrics).reportRoutingSuccessQps(0);
    }

    @Test
    void processRequest_shouldRetryOnRetryableError() throws Exception {
        BalanceContext ctx = createContext("request-1");
        Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        when(router.route(ctx)).thenReturn(Mono.just(errorResponse), Mono.just(successResponse));

        scheduler.processRequest(ctx).block();

        assertEquals(1, ctx.getRetryCount());
        assertTrue(ctx.getFuture().get().isSuccess());
        verify(router, times(2)).route(ctx);
        verify(metrics).reportRoutingFailureQps(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode());
    }

    @Test
    void processRequest_shouldNotRetryOnNonRetryableError() throws Exception {
        BalanceContext ctx = createContext("request-1");
        Response errorResponse = Response.error(StrategyErrorType.INVALID_REQUEST);
        when(router.route(ctx)).thenReturn(Mono.just(errorResponse));

        scheduler.processRequest(ctx).block();

        assertEquals(0, ctx.getRetryCount());
        assertTrue(ctx.getFuture().isDone());
        assertFalse(ctx.getFuture().get().isSuccess());
    }

    @Test
    void processRequest_shouldStopRetryingAfterMaxRetries() throws Exception {
        BalanceContext ctx = createContext("request-1");
        Response errorResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        when(router.route(ctx)).thenReturn(Mono.just(errorResponse));

        scheduler.processRequest(ctx).block();

        assertEquals(3, ctx.getRetryCount());
        verify(router, times(4)).route(ctx);
        assertTrue(ctx.getFuture().isDone());
        assertFalse(ctx.getFuture().get().isSuccess());
        assertEquals(StrategyErrorType.NO_AVAILABLE_WORKER.getErrorCode(), ctx.getFuture().get().getCode());
    }

    @Test
    void processRequest_shouldCompleteExceptionallyOnException() throws Exception {
        BalanceContext ctx = createContext("request-1");
        when(router.route(ctx)).thenReturn(Mono.error(new RuntimeException("routing error")));

        scheduler.processRequest(ctx).block();

        assertTrue(ctx.getFuture().isCompletedExceptionally());
    }

    @Test
    void processRequest_shouldRollBackSuccessfulRouteWhenContextIsCancelledBeforeResponseHandling() {
        BalanceContext ctx = createContext("request-cancelled-after-route");
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        when(router.route(ctx)).thenReturn(Mono.just(successResponse).doOnNext(ignored -> ctx.cancel()));

        scheduler.processRequest(ctx).block();

        verify(router).rollBack(ctx, successResponse);
        assertTrue(ctx.getFuture().isCompletedExceptionally());
    }

    @Test
    void processRequest_shouldCompleteFutureExceptionallyWhenRouterCompletesEmpty() {
        BalanceContext ctx = createContext("request-empty-route");
        when(router.route(ctx)).thenReturn(Mono.empty());

        scheduler.processRequest(ctx).block();

        assertTrue(ctx.getFuture().isCompletedExceptionally());
    }

    @Test
    void processRequest_shouldRetryWithoutRecursiveStackGrowthWhenRetryIsUnlimited() {
        BalanceContext ctx = createContext("request-unlimited-retry");
        ctx.getConfig().setMaxRetryCount(0);
        Response retryableResponse = Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        AtomicInteger attempts = new AtomicInteger();
        when(router.route(ctx)).thenAnswer(invocation -> Mono.just(
                attempts.incrementAndGet() <= 128 ? retryableResponse : successResponse));

        scheduler.processRequest(ctx).block(Duration.ofSeconds(2));

        assertEquals(129, attempts.get());
        assertEquals(128, ctx.getRetryCount());
        assertTrue(ctx.getFuture().isDone());
        assertTrue(ctx.getFuture().join().isSuccess());
    }

    @Test
    void workerLoop_shouldKeepPermitUntilAsyncRouteCompletes() throws Exception {
        BalanceContext ctx = createContext("request-async");
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        Sinks.One<Response> pendingRoute = Sinks.one();
        CountDownLatch workerReturnedToPermitLoop = new CountDownLatch(1);
        AtomicInteger acquireAttempts = new AtomicInteger();
        when(dynamicWorkerManager.tryAcquirePermit(500, TimeUnit.MILLISECONDS)).thenAnswer(invocation -> {
            if (acquireAttempts.incrementAndGet() == 1) {
                return true;
            }
            workerReturnedToPermitLoop.countDown();
            TimeUnit.MILLISECONDS.sleep(10);
            return false;
        });
        when(queueManager.takeRequest(true, 500)).thenReturn(ctx);
        when(router.route(ctx)).thenReturn(pendingRoute.asMono());

        scheduler.start();

        assertTrue(workerReturnedToPermitLoop.await(1, TimeUnit.SECONDS));
        verify(dynamicWorkerManager, never()).releasePermit();

        pendingRoute.tryEmitValue(successResponse);

        verify(dynamicWorkerManager, timeout(1_000)).releasePermit();
        assertTrue(ctx.getFuture().get(1, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void workerLoop_shouldCancelInFlightRouteWhenContextIsCancelled() throws Exception {
        BalanceContext ctx = createContext("request-cancelled");
        Response successResponse = new Response();
        successResponse.setSuccess(true);
        Sinks.One<Response> pendingRoute = Sinks.one();
        CountDownLatch routeCancelled = new CountDownLatch(1);
        CountDownLatch workerReturnedToPermitLoop = new CountDownLatch(1);
        AtomicInteger acquireAttempts = new AtomicInteger();
        when(dynamicWorkerManager.tryAcquirePermit(500, TimeUnit.MILLISECONDS)).thenAnswer(invocation -> {
            if (acquireAttempts.incrementAndGet() == 1) {
                return true;
            }
            workerReturnedToPermitLoop.countDown();
            TimeUnit.MILLISECONDS.sleep(10);
            return false;
        });
        when(queueManager.takeRequest(true, 500)).thenReturn(ctx);
        when(router.route(ctx)).thenReturn(pendingRoute.asMono().doOnCancel(routeCancelled::countDown));

        scheduler.start();

        assertTrue(workerReturnedToPermitLoop.await(1, TimeUnit.SECONDS));
        ctx.cancel();

        verify(dynamicWorkerManager, timeout(1_000)).releasePermit();
        assertTrue(routeCancelled.await(1, TimeUnit.SECONDS));
    }

    private BalanceContext createContext(String requestId) {
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
