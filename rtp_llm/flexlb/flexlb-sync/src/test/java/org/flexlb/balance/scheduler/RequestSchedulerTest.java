package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFallback;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.flexlb.dao.route.RoleType;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyBoolean;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.when;

class RequestSchedulerTest {

    @Test
    void terminalRouteRejectionDoesNotAcquireDecodeAcceptance() {
        Fixture fixture = new Fixture(true);

        Response response = fixture.scheduler.submit(fixture.context).join();

        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                response.getCode());
        verify(fixture.router, timeout(1_000))
                .routeForQueue(fixture.context);
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), anyBoolean(), anyInt(), anyLong(), any());
        fixture.scheduler.closePlacement();
    }

    @Test
    void priorityTemporaryMissInvokesFallbackAndKeepsOriginalFuture() {
        Fixture fixture = new Fixture(true);
        when(fixture.router.routeForQueue(fixture.context)).thenReturn(
                new QueueRoutingResult.Blocked(
                        PlacementKey.anyGroup(RoleType.PREFILL), true));
        when(fixture.lifecycle.isAdmissionOpen(
                fixture.requestId, fixture.future)).thenReturn(true);
        when(fixture.fallback.tryAdmit(
                fixture.context, fixture.future)).thenReturn(false);

        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.fallback, timeout(1_000))
                .tryAdmit(fixture.context, fixture.future);
        assertFalse(waiting.isDone());
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), anyBoolean(), anyInt(), anyLong(), any());
        fixture.scheduler.closePlacement();
    }

    @Test
    void fifoTemporaryMissWaitsWithoutFallbackOrAcceptanceGuard() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.routeForQueue(fixture.context)).thenReturn(
                new QueueRoutingResult.Blocked(
                        PlacementKey.anyGroup(RoleType.DECODE), true));
        when(fixture.lifecycle.isAdmissionOpen(
                fixture.requestId, fixture.future)).thenReturn(true);

        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.router, timeout(1_000))
                .routeForQueue(fixture.context);
        assertFalse(waiting.isDone());
        verify(fixture.fallback, never()).tryAdmit(any(), any());
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), anyBoolean(), anyInt(), anyLong(), any());
        fixture.scheduler.closePlacement();
    }

    private static final class Fixture {
        private final long requestId = 701L;
        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        private final ConfigService configService = mock(ConfigService.class);
        private final Router router = mock(Router.class);
        private final AdmissionFallback fallback = mock(AdmissionFallback.class);
        private final RequestLifecycleCoordinator lifecycle =
                mock(RequestLifecycleCoordinator.class);
        private final BalanceContext context = mock(BalanceContext.class);
        private final CompletableFuture<Response> future =
                new CompletableFuture<>();
        private final RequestScheduler scheduler;

        private Fixture(boolean priority) {
            if (priority) {
                SchedulingTestConfig.usePriorityQueue(config);
            } else {
                SchedulingTestConfig.useFifoQueue(config);
            }
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(context.getRequest()).thenReturn(new Request());
            when(context.getRequestId()).thenReturn(requestId);
            when(lifecycle.register(context, config.queueScheduler()
                    .getCapacity().getMaxOutstandingRequestsGlobal()))
                    .thenReturn(future);
            when(lifecycle.beginAdmission(requestId, future)).thenReturn(
                    mock(RequestLifecycleCoordinator.AdmissionScope.class));
            when(router.routeForQueue(context)).thenReturn(
                    new QueueRoutingResult.Rejected(
                            RequestLifecycleCoordinator.buildErrorResponse(
                                    StrategyErrorType.NO_PREFILL_WORKER,
                                    null)));
            scheduler = new RequestScheduler(
                    configService,
                    router,
                    mock(EndpointRegistry.class),
                    mock(BatchSchedulerReporter.class),
                    fallback,
                    lifecycle);
        }
    }
}
