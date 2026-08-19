package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.ScheduleModeEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
    private BalanceContext balanceContext;

    private RouteService routeService;

    @BeforeEach
    void setUp() {
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
        routeService = new RouteService(configService, defaultRouter, queueManager,
                flexlbBatchScheduler, recentCacheKeyTraceReporter);
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

    @Test
    void batch_mode_should_fail_closed_when_generate_input_is_missing() {
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.BATCH);

        Response actual = routeService.route(balanceContext).join();

        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(), actual.getCode());
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(flexlbBatchScheduler, never()).submit(any(BalanceContext.class));
    }

    @Test
    void batch_mode_should_submit_valid_input_without_direct_fallback() {
        Response response = successResponse();
        CompletableFuture<Response> submitted = CompletableFuture.completedFuture(response);
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.BATCH);
        when(balanceContext.getGenerateInputPbBytes()).thenReturn(new byte[]{1});
        when(flexlbBatchScheduler.submit(balanceContext)).thenReturn(submitted);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(flexlbBatchScheduler).submit(balanceContext);
        verify(balanceContext).setFuture(submitted);
        verify(defaultRouter, never()).route(any(BalanceContext.class));
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
