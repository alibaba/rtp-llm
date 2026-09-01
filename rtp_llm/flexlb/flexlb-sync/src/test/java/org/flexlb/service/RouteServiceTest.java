package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.lenient;
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
        lenient().when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
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
    void direct_route_runs_inline_on_the_caller_thread() {
        AtomicReference<Thread> routeThread = new AtomicReference<>();
        Response response = successResponse();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.DIRECT);
        when(defaultRouter.route(balanceContext)).thenAnswer(invocation -> {
            routeThread.set(Thread.currentThread());
            return response;
        });

        Thread caller = Thread.currentThread();
        routeService.route(balanceContext).join();

        assertSame(caller, routeThread.get(),
                "direct routing commits a reservation and must not hop schedulers");
    }

    @Test
    void batch_schedule_hops_off_the_caller_thread() {
        AtomicReference<Thread> scheduleThread = new AtomicReference<>();
        BatchScheduleResponse response = org.mockito.Mockito.mock(BatchScheduleResponse.class);
        when(defaultRouter.batchSchedule(any(BatchScheduleRequest.class)))
                .thenAnswer(invocation -> {
                    scheduleThread.set(Thread.currentThread());
                    return response;
                });

        Thread caller = Thread.currentThread();
        routeService.batchSchedule(new BatchScheduleRequest()).block();

        assertNotSame(caller, scheduleThread.get(),
                "batch target selection must not block the subscribing event-loop thread");
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
