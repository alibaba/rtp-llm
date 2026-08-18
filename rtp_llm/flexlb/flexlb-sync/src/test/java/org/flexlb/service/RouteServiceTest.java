package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.balance.scheduler.QueueManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.enums.ScheduleModeEnum;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import reactor.core.publisher.Mono;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doThrow;
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
    private PriorityScheduler priorityScheduler;

    @Mock
    private RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    @Mock
    private BalanceContext balanceContext;

    private RouteService routeService;

    @BeforeEach
    void setUp() {
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
        routeService = new RouteService(configService, defaultRouter, queueManager,
                priorityScheduler, recentCacheKeyTraceReporter);
    }

    @Test
    void queue_without_auto_tpm_should_use_legacy_queue_manager() {
        Response response = successResponse();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.QUEUE);
        when(flexlbConfig.usesRouteDecisionDelivery()).thenReturn(false);
        when(queueManager.tryRouteAsync(balanceContext)).thenReturn(Mono.just(response));

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(queueManager).tryRouteAsync(balanceContext);
        verify(priorityScheduler, never()).submit(any(BalanceContext.class));
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void queue_with_auto_tpm_should_use_priority_scheduler_without_generate_input() {
        Response response = successResponse();
        CompletableFuture<Response> schedulerFuture = CompletableFuture.completedFuture(response);
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.QUEUE);
        when(flexlbConfig.usesRouteDecisionDelivery()).thenReturn(true);
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(priorityScheduler).submit(balanceContext);
        verify(balanceContext).setFuture(schedulerFuture);
        verify(balanceContext, never()).getGenerateInputPbBytes();
        verify(queueManager, never()).tryRouteAsync(any(BalanceContext.class));
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void priorityRouteReturnsSchedulerFutureSoCallerCancelReachesItsOwner() {
        CompletableFuture<Response> schedulerFuture = new CompletableFuture<>();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.QUEUE);
        when(flexlbConfig.usesRouteDecisionDelivery()).thenReturn(true);
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);

        CompletableFuture<Response> publicFuture = routeService.route(balanceContext);

        assertSame(schedulerFuture, publicFuture);
        assertTrue(publicFuture.cancel(false));
        assertTrue(schedulerFuture.isCancelled());
    }

    @Test
    void completionSideEffectFailureCannotReplaceSuccessfulSchedulerResult() {
        Response response = successResponse();
        CompletableFuture<Response> schedulerFuture = new CompletableFuture<>();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.QUEUE);
        when(flexlbConfig.usesRouteDecisionDelivery()).thenReturn(true);
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);
        doThrow(new IllegalStateException("telemetry failed"))
                .when(recentCacheKeyTraceReporter).report(balanceContext);

        CompletableFuture<Response> publicFuture = routeService.route(balanceContext);
        schedulerFuture.complete(response);

        assertSame(schedulerFuture, publicFuture);
        assertSame(response, publicFuture.join());
    }

    @Test
    void batch_with_generate_input_should_preserve_priority_scheduler_path() {
        Response response = successResponse();
        CompletableFuture<Response> schedulerFuture = CompletableFuture.completedFuture(response);
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.BATCH);
        when(balanceContext.getGenerateInputPbBytes()).thenReturn(new byte[]{1});
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(priorityScheduler).submit(balanceContext);
        verify(balanceContext).setFuture(schedulerFuture);
        verify(queueManager, never()).tryRouteAsync(any(BalanceContext.class));
        verify(defaultRouter, never()).route(any(BalanceContext.class));
    }

    @Test
    void batch_without_generate_input_should_preserve_direct_fallback() {
        Response response = successResponse();
        when(flexlbConfig.getDefaultScheduleModeEnum()).thenReturn(ScheduleModeEnum.BATCH);
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(priorityScheduler, never()).submit(any(BalanceContext.class));
        verify(balanceContext).setScheduleMode(ScheduleModeEnum.DIRECT);
        verify(defaultRouter).route(balanceContext);
        verify(queueManager, never()).tryRouteAsync(any(BalanceContext.class));
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

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
