package org.flexlb.service;

import org.flexlb.balance.scheduler.DefaultRouter;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DirectSchedulerConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.QueueSchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class RouteServiceTest {

    @Mock
    private ConfigService configService;

    private FlexlbConfig flexlbConfig;

    @Mock
    private DefaultRouter defaultRouter;

    @Mock
    private PriorityScheduler priorityScheduler;

    @Mock
    private RecentCacheKeyTraceReporter recentCacheKeyTraceReporter;

    @Mock
    private BalanceContext balanceContext;

    private RouteService routeService;

    @BeforeEach
    void setUp() {
        flexlbConfig = new FlexlbConfig();
        when(configService.loadBalanceConfig()).thenReturn(flexlbConfig);
        routeService = new RouteService(configService, defaultRouter,
                priorityScheduler, recentCacheKeyTraceReporter);
    }

    @Test
    void fifo_queue_with_non_batch_dispatch_should_use_common_scheduler() {
        Response response = successResponse();
        CompletableFuture<Response> schedulerFuture = CompletableFuture.completedFuture(response);
        useFifoNonBatch();
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(priorityScheduler).submit(balanceContext);
        verify(balanceContext).setFuture(schedulerFuture);
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void priority_queue_with_non_batch_dispatch_should_use_common_scheduler_without_generate_input() {
        Response response = successResponse();
        CompletableFuture<Response> schedulerFuture = CompletableFuture.completedFuture(response);
        usePriorityNonBatch();
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(priorityScheduler).submit(balanceContext);
        verify(balanceContext).setFuture(schedulerFuture);
        verify(balanceContext, never()).getGenerateInputPbBytes();
        verify(defaultRouter, never()).route(any(BalanceContext.class));
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void queueRouteReturnsSchedulerFutureSoCallerCancelReachesItsOwner() {
        CompletableFuture<Response> schedulerFuture = new CompletableFuture<>();
        usePriorityNonBatch();
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
        usePriorityNonBatch();
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
        useBatch();
        when(balanceContext.getGenerateInputPbBytes()).thenReturn(new byte[]{1});
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(priorityScheduler).submit(balanceContext);
        verify(balanceContext).setFuture(schedulerFuture);
        verify(defaultRouter, never()).route(any(BalanceContext.class));
    }

    @Test
    void priority_batch_dispatch_should_use_same_common_scheduler() {
        Response response = successResponse();
        CompletableFuture<Response> schedulerFuture = CompletableFuture.completedFuture(response);
        usePriorityBatch();
        when(balanceContext.getGenerateInputPbBytes()).thenReturn(new byte[]{1});
        when(priorityScheduler.submit(balanceContext)).thenReturn(schedulerFuture);

        assertSame(response, routeService.route(balanceContext).join());

        verify(priorityScheduler).submit(balanceContext);
        verify(balanceContext).setFuture(schedulerFuture);
        verify(defaultRouter, never()).route(any(BalanceContext.class));
    }

    @Test
    void batch_without_generate_input_should_fail_instead_of_changing_delivery_protocol() {
        useBatch();

        Response actual = routeService.route(balanceContext).join();

        assertFalse(actual.isSuccess());
        assertEquals(StrategyErrorType.BATCH_BUILD_FAILED.getErrorCode(), actual.getCode());
        verify(priorityScheduler, never()).submit(any(BalanceContext.class));
        verify(defaultRouter, never()).route(any(BalanceContext.class));
    }

    @Test
    void should_report_recent_cache_key_once_after_direct_route_success() {
        Response response = successResponse();
        useDirect();
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(defaultRouter).route(balanceContext);
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter).report(balanceContext);
    }

    @Test
    void expired_direct_request_fails_before_worker_reservation() {
        useDirect();
        when(balanceContext.requestExpired(anyLong())).thenReturn(true);

        Response actual = routeService.route(balanceContext).join();

        assertFalse(actual.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), actual.getCode());
        verify(defaultRouter, never()).route(any(BalanceContext.class));
    }

    @Test
    void should_not_report_recent_cache_key_after_route_failure() {
        Response response = new Response();
        response.setSuccess(false);
        useDirect();
        when(defaultRouter.route(balanceContext)).thenReturn(response);

        Response actual = routeService.route(balanceContext).join();

        assertSame(response, actual);
        verify(balanceContext).setConfig(flexlbConfig);
        verify(defaultRouter).route(balanceContext);
        verify(balanceContext).setResponse(response);
        verify(recentCacheKeyTraceReporter, never()).report(any(BalanceContext.class));
    }

    private void useFifoNonBatch() {
        flexlbConfig.setScheduler(new QueueSchedulerConfig());
        flexlbConfig.setDispatcher(new NonBatchDispatcherConfig());
        when(balanceContext.getConfig()).thenReturn(flexlbConfig);
    }

    private void usePriorityNonBatch() {
        QueueSchedulerConfig scheduler = new QueueSchedulerConfig();
        scheduler.setOrdering(new PriorityOrderingConfig());
        flexlbConfig.setScheduler(scheduler);
        flexlbConfig.setDispatcher(new NonBatchDispatcherConfig());
        when(balanceContext.getConfig()).thenReturn(flexlbConfig);
    }

    private void useBatch() {
        flexlbConfig.setScheduler(new QueueSchedulerConfig());
        flexlbConfig.setDispatcher(new BatchDispatcherConfig());
        when(balanceContext.getConfig()).thenReturn(flexlbConfig);
    }

    private void usePriorityBatch() {
        QueueSchedulerConfig scheduler = new QueueSchedulerConfig();
        scheduler.setOrdering(new PriorityOrderingConfig());
        flexlbConfig.setScheduler(scheduler);
        flexlbConfig.setDispatcher(new BatchDispatcherConfig());
        when(balanceContext.getConfig()).thenReturn(flexlbConfig);
    }

    private void useDirect() {
        flexlbConfig.setScheduler(new DirectSchedulerConfig());
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
