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
import org.mockito.InOrder;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestSchedulerTest {

    @Test
    void priorityBindsDecodeAcceptanceBeforeRouting() {
        Fixture fixture = new Fixture(true);
        when(fixture.lifecycle.tryInstallDecodeAcceptanceGuard(
                anyLong(), any(), anyInt(), anyLong())).thenReturn(true);

        fixture.scheduler.submit(fixture.context);

        InOrder order = inOrder(fixture.lifecycle, fixture.router);
        order.verify(fixture.lifecycle).beginAdmission(
                fixture.requestId, fixture.future);
        order.verify(fixture.lifecycle).tryInstallDecodeAcceptanceGuard(
                fixture.requestId,
                fixture.future,
                fixture.acceptanceLimit,
                fixture.acceptanceTimeoutMs);
        order.verify(fixture.router).routeForQueue(fixture.context);
    }

    @Test
    void priorityCapacityRejectionDoesNotPublishAQueueRoute() {
        Fixture fixture = new Fixture(true);
        when(fixture.lifecycle.tryInstallDecodeAcceptanceGuard(
                anyLong(), any(), anyInt(), anyLong())).thenReturn(false);
        when(fixture.lifecycle.decodeAcceptanceCount())
                .thenReturn(fixture.acceptanceLimit);

        Response response = fixture.scheduler.submit(fixture.context).join();

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                response.getCode());
        verify(fixture.router, never()).routeForQueue(any());
    }

    @Test
    void fifoAlsoInstallsDecodeAcceptanceGuardBeforeRouting() {
        Fixture fixture = new Fixture(false);
        when(fixture.lifecycle.tryInstallDecodeAcceptanceGuard(
                anyLong(), any(), anyInt(), anyLong())).thenReturn(true);

        fixture.scheduler.submit(fixture.context);

        InOrder order = inOrder(fixture.lifecycle, fixture.router);
        order.verify(fixture.lifecycle).tryInstallDecodeAcceptanceGuard(
                fixture.requestId,
                fixture.future,
                fixture.acceptanceLimit,
                fixture.acceptanceTimeoutMs);
        order.verify(fixture.router).routeForQueue(fixture.context);
    }

    private static final class Fixture {
        private final long requestId = 701L;
        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        private final int acceptanceLimit;
        private final long acceptanceTimeoutMs;
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
            acceptanceLimit = config.queueScheduler().getLifecycle()
                    .getMaxDeliveredNotAcceptedRequestsGlobal();
            acceptanceTimeoutMs = config.queueScheduler().getLifecycle()
                    .getDeliveredNotAcceptedTimeoutMs();
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
