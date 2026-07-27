package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
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
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class QueueManagerTest {

    @Mock
    private RoutingQueueReporter metrics;
    @Mock
    private ConfigService configService;
    @Mock
    private EndpointRegistry endpointRegistry;

    private QueueManager queueManager;

    @BeforeEach
    void setUp() {
        FlexlbConfig config = new FlexlbConfig();
        config.setMaxQueueSize(10);
        when(configService.loadBalanceConfig()).thenReturn(config);
        queueManager = new QueueManager(metrics, configService, endpointRegistry);
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
        BalanceContext result = queueManager.takeRequest(0);
        assertNull(result);
    }

    @Test
    void takeRequest_shouldReturnEnqueuedRequest() {
        BalanceContext ctx = createContext(1L);
        queueManager.tryRouteAsync(ctx);

        BalanceContext taken = queueManager.takeRequest(0);
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

        BalanceContext taken = queueManager.takeRequest(0);
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

        BalanceContext taken = queueManager.takeRequest(0);
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

    // ==================== cancel() tests ====================

    @Test
    void cancel_completesFutureExceptionally() {
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(new ConcurrentHashMap<>());

        BalanceContext ctx = createContext(1L);
        queueManager.tryRouteAsync(ctx);
        queueManager.cancel(ctx);

        assertTrue(ctx.getFuture().isDone());
        assertThrows(CancellationException.class, () -> ctx.getFuture().join());
    }

    @Test
    void cancel_releasesPrefillInflightViaCallback() {
        AtomicBoolean callbackInvoked = new AtomicBoolean(false);
        PrefillEndpoint mockEp = Mockito.mock(PrefillEndpoint.class);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());
        Mockito.lenient().when(endpointRegistry.getPrefillEndpoints())
                .thenReturn(new ConcurrentHashMap<>(Map.of("ep1", mockEp)));

        BalanceContext ctx = createContext(1L);
        ctx.setPrefillReleaseCallback(() -> callbackInvoked.set(true));
        queueManager.cancel(ctx);

        assertTrue(callbackInvoked.get());
        verify(mockEp, never()).releaseBatch(1L);
    }

    @Test
    void cancel_releasesPrefillInflightViaBruteForce_whenCallbackNull() {
        PrefillEndpoint mockEp = Mockito.mock(PrefillEndpoint.class);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());
        when(endpointRegistry.getPrefillEndpoints())
                .thenReturn(new ConcurrentHashMap<>(Map.of("ep1", mockEp)));

        BalanceContext ctx = createContext(1L);
        queueManager.cancel(ctx);

        verify(mockEp).releaseBatch(1L);
    }

    @Test
    void cancel_releasesDecodeInflightViaBruteForce() {
        DecodeEndpoint mockEp = Mockito.mock(DecodeEndpoint.class);
        when(endpointRegistry.getDecodeEndpoints())
                .thenReturn(new ConcurrentHashMap<>(Map.of("ep1", mockEp)));
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(new ConcurrentHashMap<>());

        BalanceContext ctx = createContext(1L);
        queueManager.cancel(ctx);

        verify(mockEp).release(1L);
    }

    @Test
    void cancel_isIdempotent_multipleCallsNoError() {
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(new ConcurrentHashMap<>());

        BalanceContext ctx = createContext(1L);
        queueManager.cancel(ctx);
        assertDoesNotThrow(() -> queueManager.cancel(ctx));
    }

    // ==================== cancelByRequestId() tests ====================

    @Test
    void cancelByRequestId_releasesAllPrefillEndpoints() {
        PrefillEndpoint mockEp1 = Mockito.mock(PrefillEndpoint.class);
        PrefillEndpoint mockEp2 = Mockito.mock(PrefillEndpoint.class);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());
        when(endpointRegistry.getPrefillEndpoints())
                .thenReturn(new ConcurrentHashMap<>(Map.of("ep1", mockEp1, "ep2", mockEp2)));

        queueManager.cancelByRequestId(1L);

        verify(mockEp1).releaseBatch(1L);
        verify(mockEp2).releaseBatch(1L);
    }

    @Test
    void cancelByRequestId_releasesAllDecodeEndpoints() {
        DecodeEndpoint mockEp1 = Mockito.mock(DecodeEndpoint.class);
        DecodeEndpoint mockEp2 = Mockito.mock(DecodeEndpoint.class);
        when(endpointRegistry.getDecodeEndpoints())
                .thenReturn(new ConcurrentHashMap<>(Map.of("ep1", mockEp1, "ep2", mockEp2)));
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(new ConcurrentHashMap<>());

        queueManager.cancelByRequestId(1L);

        verify(mockEp1).release(1L);
        verify(mockEp2).release(1L);
    }

    @Test
    void cancelByRequestId_safeWhenNoEndpointsRegistered() {
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(new ConcurrentHashMap<>());

        assertDoesNotThrow(() -> queueManager.cancelByRequestId(1L));
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
