package org.flexlb.service;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.DynamicWorkerManager;
import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.balance.scheduler.RouteResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RouteServiceTest {

    @Mock
    private ConfigService configService;

    @Mock
    private DefaultRouter defaultRouter;

    @Mock
    private RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    @Mock
    private BatchSchedulerReporter batchSchedulerReporter;

    @Mock
    private RoutingQueueReporter routingQueueReporter;

    @Mock
    private DynamicWorkerManager dynamicWorkerManager;

    @Mock
    private EndpointRegistry endpointRegistry;

    private FlexlbConfig flexlbConfig;
    private RouteService routeService;

    @BeforeEach
    void setUp() {
        flexlbConfig = new FlexlbConfig();
        flexlbConfig.setScheduleWorkerSize(1);
        flexlbConfig.setQueueingComponentQueueMaxSize(10);
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
        routeService = new RouteService(configService, defaultRouter,
                recentCacheKeyTraceReporter, NoOpFlexMonitor.getInstance(),
                new InflightStore(batchSchedulerReporter, configService),
                endpointRegistry, batchSchedulerReporter,
                routingQueueReporter, dynamicWorkerManager);
    }

    @AfterEach
    void tearDown() {
        routeService.shutdown();
    }

    @Test
    void should_report_recent_cache_key_once_after_queued_route_success() throws Exception {
        RouteResult routeResult = RouteResult.success(null, null, List.of());
        flexlbConfig.setDefaultScheduleMode("QUEUE");
        lenient().when(dynamicWorkerManager.tryAcquirePermit(anyLong(), any())).thenReturn(true);
        when(defaultRouter.route(any(BalanceContext.class))).thenReturn(routeResult);
        routeService.start(); // start the queue consumer workers

        BalanceContext balanceContext = createContext(1L);
        Response actual = routeService.route(balanceContext).get(5, TimeUnit.SECONDS);

        assertTrue(actual.isSuccess());
        assertSame(flexlbConfig, balanceContext.getConfig());
        verify(defaultRouter).route(balanceContext);
        assertTrue(balanceContext.getResponse().isSuccess());
        verify(recentCacheKeyTraceReporter, timeout(1000)).report(balanceContext);
    }

    @Test
    void should_report_recent_cache_key_once_after_direct_route_success() throws Exception {
        RouteResult routeResult = RouteResult.success(null, null, List.of());
        flexlbConfig.setDefaultScheduleMode("DIRECT");
        when(defaultRouter.route(any(BalanceContext.class))).thenReturn(routeResult);

        BalanceContext balanceContext = createContext(2L);
        Response actual = routeService.route(balanceContext).get(5, TimeUnit.SECONDS);

        assertTrue(actual.isSuccess());
        assertSame(flexlbConfig, balanceContext.getConfig());
        verify(defaultRouter).route(balanceContext);
        assertTrue(balanceContext.getResponse().isSuccess());
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void should_not_report_recent_cache_key_after_route_failure() throws Exception {
        RouteResult failure = RouteResult.failure(StrategyErrorType.NO_AVAILABLE_WORKER, "test failure");
        flexlbConfig.setDefaultScheduleMode("DIRECT");
        when(defaultRouter.route(any(BalanceContext.class))).thenReturn(failure);

        BalanceContext balanceContext = createContext(3L);
        Response actual = routeService.route(balanceContext).get(5, TimeUnit.SECONDS);

        assertFalse(actual.isSuccess());
        assertFalse(balanceContext.getResponse().isSuccess());
        verify(recentCacheKeyTraceReporter, never()).report(any(BalanceContext.class));
    }

    private static BalanceContext createContext(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setGenerateTimeout(60_000);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        return ctx;
    }
}
