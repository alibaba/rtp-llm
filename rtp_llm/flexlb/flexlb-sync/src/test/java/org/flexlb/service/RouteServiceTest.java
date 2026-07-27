package org.flexlb.service;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.balance.scheduler.RequestLifecycleSnapshot;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RouteServiceTest {

    @Mock
    private ConfigService configService;

    @Mock
    private FlexlbConfig flexlbConfig;

    @Mock
    private DefaultRouter defaultRouter;

    @Mock
    private QueueManager queueManager;

    @Mock
    private org.flexlb.balance.scheduler.FlexlbBatchScheduler flexlbBatchScheduler;

    @Mock
    private RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    @Mock
    private EndpointRegistry endpointRegistry;

    @Mock
    private BalanceContext balanceContext;

    private RouteService routeService;

    @BeforeEach
    void setUp() {
        Mockito.lenient().when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
        routeService = new RouteService(configService, defaultRouter, queueManager,
                flexlbBatchScheduler, recentCacheKeyTraceReporter,
                endpointRegistry);
    }

    @Test
    void should_report_recent_cache_key_once_after_queued_route_success() {
        Response response = successResponse();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.QUEUE);
        when(queueManager.tryRouteAsync(balanceContext)).thenReturn(Mono.just(response));

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(queueManager).tryRouteAsync(balanceContext);
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void should_report_recent_cache_key_once_after_direct_route_success() {
        Response response = successResponse();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.DIRECT);
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(defaultRouter).route(balanceContext);
        verify(queueManager, never()).tryRouteAsync(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void should_not_report_recent_cache_key_after_route_failure() {
        Response response = new Response();
        response.setSuccess(false);
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.DIRECT);
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(defaultRouter).route(balanceContext);
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter, never()).report(any(BalanceContext.class));
    }

    // ==================== cancel() tests ====================

    @Test
    void cancel_queueMode_delegatesToQueueManager() {
        when(balanceContext.getScheduleMode()).thenReturn(ScheduleModeEnum.QUEUE);

        routeService.cancel(balanceContext, CancelReason.CLIENT_CANCELLED);

        verify(queueManager).cancel(balanceContext);
        verify(flexlbBatchScheduler, never()).cancel(anyLong(), any(), anyLong());
    }

    @Test
    void cancel_batchMode_delegatesToBatchScheduler() {
        Request mockRequest = Mockito.mock(Request.class);
        when(balanceContext.getScheduleMode()).thenReturn(ScheduleModeEnum.BATCH);
        when(balanceContext.getRequest()).thenReturn(mockRequest);
        when(mockRequest.getRequestId()).thenReturn(1L);

        routeService.cancel(balanceContext, CancelReason.CLIENT_CANCELLED);

        verify(flexlbBatchScheduler).cancel(1L, CancelReason.CLIENT_CANCELLED, 0);
        verify(queueManager, never()).cancel(balanceContext);
    }

    @Test
    void cancel_directMode_releasesDecodeInflightViaBruteForce() {
        DecodeEndpoint mockDecodeEp = Mockito.mock(DecodeEndpoint.class);
        when(balanceContext.getScheduleMode()).thenReturn(ScheduleModeEnum.DIRECT);
        when(balanceContext.getRequestId()).thenReturn(1L);
        when(endpointRegistry.getDecodeEndpoints())
                .thenReturn(new ConcurrentHashMap<>(Map.of("ep1", mockDecodeEp)));
        when(endpointRegistry.getPrefillEndpoints()).thenReturn(new ConcurrentHashMap<>());

        routeService.cancel(balanceContext, CancelReason.CLIENT_CANCELLED);

        verify(mockDecodeEp).release(1L);
    }

    @Test
    void cancel_directMode_releasesPrefillInflightViaCallback() {
        Runnable mockRunnable = Mockito.mock(Runnable.class);
        when(balanceContext.getScheduleMode()).thenReturn(ScheduleModeEnum.DIRECT);
        when(balanceContext.getPrefillReleaseCallback()).thenReturn(mockRunnable);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());

        routeService.cancel(balanceContext, CancelReason.CLIENT_CANCELLED);

        verify(mockRunnable).run();
    }

    @Test
    void cancel_directMode_releasesPrefillInflightViaBruteForce_whenCallbackNull() {
        PrefillEndpoint mockPrefillEp = Mockito.mock(PrefillEndpoint.class);
        when(balanceContext.getScheduleMode()).thenReturn(ScheduleModeEnum.DIRECT);
        when(balanceContext.getRequestId()).thenReturn(1L);
        when(endpointRegistry.getDecodeEndpoints()).thenReturn(new ConcurrentHashMap<>());
        when(endpointRegistry.getPrefillEndpoints())
                .thenReturn(new ConcurrentHashMap<>(Map.of("ep1", mockPrefillEp)));

        routeService.cancel(balanceContext, CancelReason.CLIENT_CANCELLED);

        verify(mockPrefillEp).releaseBatch(1L);
    }

    // ==================== cancelByRequestId() tests ====================

    @Test
    void cancelByRequestId_delegatesToBatchScheduler_whenInBatchInflight() {
        RequestLifecycleSnapshot snapshot = Mockito.mock(RequestLifecycleSnapshot.class);
        when(flexlbBatchScheduler.cancel(1L, CancelReason.CLIENT_CANCELLED, 0)).thenReturn(snapshot);

        RequestLifecycleSnapshot result = routeService.cancelByRequestId(1L, CancelReason.CLIENT_CANCELLED, 0);

        assertSame(snapshot, result);
        verify(queueManager, never()).cancelByRequestId(1L);
    }

    @Test
    void cancelByRequestId_fallsBackToQueueManager_whenNotInBatchInflight() {
        when(flexlbBatchScheduler.cancel(1L, CancelReason.CLIENT_CANCELLED, 0)).thenReturn(null);

        RequestLifecycleSnapshot result = routeService.cancelByRequestId(1L, CancelReason.CLIENT_CANCELLED, 0);

        assertNull(result);
        verify(queueManager).cancelByRequestId(1L);
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
