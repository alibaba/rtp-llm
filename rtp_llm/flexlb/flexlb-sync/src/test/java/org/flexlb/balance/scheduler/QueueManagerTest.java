package org.flexlb.balance.scheduler;

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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class QueueManagerTest {

    @Mock
    private RoutingQueueReporter metrics;
    @Mock
    private ConfigService configService;

    private QueueManager queueManager;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = new FlexlbConfig();
        config.setMaxQueueSize(10);
        when(configService.loadBalanceConfig()).thenReturn(config);
        queueManager = new QueueManager(metrics, configService);
    }

    @Test
    void tryRouteAsync_shouldEnqueueSuccessfully() {
        BalanceContext ctx = createContext(1L);
        var mono = queueManager.tryRouteAsync(ctx);

        assertNotNull(mono);
        assertNotNull(ctx.getFuture());
        assertTrue(ctx.getEnqueueTime() > 0);
        verify(metrics).reportQueueEntry();
    }

    @Test
    void tryRouteAsync_shouldRejectWhenQueueFull() {
        // Fill the queue
        for (int i = 0; i < 10; i++) {
            queueManager.tryRouteAsync(createContext(i));
        }

        // 11th request should be rejected
        BalanceContext ctx = createContext(11L);
        Response response = queueManager.tryRouteAsync(ctx).block();

        assertNotNull(response);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
        verify(metrics).reportRejected();
    }

    @Test
    void takeRequest_shouldReturnNullWhenEmpty() {
        BalanceContext result = queueManager.takeRequest(false, 0);
        assertNull(result);
    }

    @Test
    void takeRequest_shouldReturnEnqueuedRequest() {
        BalanceContext ctx = createContext(1L);
        queueManager.tryRouteAsync(ctx);

        BalanceContext taken = queueManager.takeRequest(false, 0);
        assertNotNull(taken);
        assertEquals(1L, taken.getRequestId());
    }

    @Test
    void takeRequest_shouldSkipCancelledRequests() {
        BalanceContext cancelled = createContext(1L);
        queueManager.tryRouteAsync(cancelled);
        cancelled.cancel();

        BalanceContext valid = createContext(2L);
        queueManager.tryRouteAsync(valid);

        BalanceContext taken = queueManager.takeRequest(false, 0);
        assertNotNull(taken);
        assertEquals(2L, taken.getRequestId());
    }

    @Test
    void offerToHead_shouldRequeueAtFront() {
        BalanceContext first = createContext(1L);
        queueManager.tryRouteAsync(first);

        BalanceContext retried = createContext(2L);
        retried.setFuture(new CompletableFuture<>());
        retried.setEnqueueTime(System.currentTimeMillis());
        queueManager.offerToHead(retried);

        BalanceContext taken = queueManager.takeRequest(false, 0);
        assertNotNull(taken);
        assertEquals(2L, taken.getRequestId());
    }

    @Test
    void offerToHead_shouldCompleteWithErrorWhenQueueFull() {
        // Fill the queue
        for (int i = 0; i < 10; i++) {
            queueManager.tryRouteAsync(createContext(i));
        }

        BalanceContext ctx = createContext(99L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        ctx.setFuture(future);

        queueManager.offerToHead(ctx);

        assertTrue(future.isDone());
        Response response = future.join();
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.QUEUE_FULL.getErrorCode(), response.getCode());
    }

    private BalanceContext createContext(long requestId) {
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setGenerateTimeout(60_000);
        ctx.setRequest(request);
        return ctx;
    }
}
