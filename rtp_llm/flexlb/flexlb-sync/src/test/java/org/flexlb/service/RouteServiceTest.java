package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import static org.junit.jupiter.api.Assertions.assertSame;
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
    private RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    @Mock
    private BalanceContext balanceContext;

    private RouteService routeService;

    @BeforeEach
    void setUp() {
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
        routeService = new RouteService(configService, defaultRouter, queueManager, recentCacheKeyTraceReporter);
    }

    @Test
    void should_report_recent_cache_key_once_after_queued_route_success() {
        Response response = successResponse();
        when(flexlbConfig.isEnableQueueing()).thenReturn(true);
        when(queueManager.tryRouteAsync(balanceContext)).thenReturn(Mono.just(response));

        Response actual = routeService.route(balanceContext).block();

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
        when(flexlbConfig.isEnableQueueing()).thenReturn(false);
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).block();

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
        when(flexlbConfig.isEnableQueueing()).thenReturn(false);
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).block();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(defaultRouter).route(balanceContext);
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter, never()).report(any(BalanceContext.class));
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
