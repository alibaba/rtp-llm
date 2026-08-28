package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
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
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.when;

class RequestSchedulerTest {

    @Test
    void expiredHeadDoesNotConsumeTheCapacityOpportunity() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mock(DefaultRouter.class);
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey blocker = PlacementKey.anyGroup(RoleType.PREFILL);

        BalanceContext expired = context(801L);
        BalanceContext follower = context(802L);
        CompletableFuture<Response> expiredFuture = new CompletableFuture<>();
        CompletableFuture<Response> followerFuture = new CompletableFuture<>();
        int maximum = config.queueScheduler().getCapacity()
                .getMaxOutstandingRequestsGlobal();
        when(lifecycle.register(expired, maximum)).thenReturn(expiredFuture);
        when(lifecycle.register(follower, maximum)).thenReturn(followerFuture);
        when(lifecycle.claimAdmissionMutation(801L, expiredFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(lifecycle.claimAdmissionMutation(802L, followerFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(lifecycle.isAdmissionOpen(801L, expiredFuture)).thenReturn(true);
        when(lifecycle.isAdmissionOpen(802L, followerFuture)).thenReturn(true);
        when(router.routeForQueue(any())).thenReturn(
                PlacementResult.blocked(blocker));

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class),
                mock(EvictionManager.class),
                lifecycle,
                availability);
        scheduler.submit(expired);
        scheduler.submit(follower);

        expired.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() - 1L));
        availability.capacityChanged(blocker);

        verify(router, timeout(1_000).times(2)).routeForQueue(follower);
        verify(router, times(1)).routeForQueue(expired);
        scheduler.closePlacement();
    }

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
                PlacementResult.blocked(
                        PlacementKey.anyGroup(RoleType.PREFILL)));
        when(fixture.lifecycle.isAdmissionOpen(
                fixture.requestId, fixture.future)).thenReturn(true);
        when(fixture.evictionManager.tryAdmit(
                fixture.context, fixture.future)).thenReturn(false);

        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.evictionManager, timeout(1_000))
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
                PlacementResult.blocked(
                        PlacementKey.anyGroup(RoleType.DECODE)));
        when(fixture.lifecycle.isAdmissionOpen(
                fixture.requestId, fixture.future)).thenReturn(true);

        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.router, timeout(1_000))
                .routeForQueue(fixture.context);
        assertFalse(waiting.isDone());
        verify(fixture.evictionManager, never()).tryAdmit(any(), any());
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), anyBoolean(), anyInt(), anyLong(), any());
        fixture.scheduler.closePlacement();
    }

    @Test
    void initialPlacementFailureCompletesTheRegisteredGeneration() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.routeForQueue(fixture.context))
                .thenThrow(new IllegalStateException("selector failed"));

        CompletableFuture<Response> returned =
                fixture.scheduler.submit(fixture.context);

        assertEquals(fixture.future, returned);
        assertEquals(StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode(),
                returned.join().getCode());
        fixture.scheduler.closePlacement();
    }

    private static BalanceContext context(long requestId) {
        BalanceContext context = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setPriority(50);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
        return context;
    }

    private static final class Fixture {
        private final long requestId = 701L;
        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        private final ConfigService configService = mock(ConfigService.class);
        private final DefaultRouter router = mock(DefaultRouter.class);
        private final EvictionManager evictionManager = mock(EvictionManager.class);
        private final RequestRegistry lifecycle =
                mock(RequestRegistry.class);
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
            when(lifecycle.claimAdmissionMutation(requestId, future)).thenReturn(
                    mock(AdmissionMutation.class));
            when(router.routeForQueue(context)).thenReturn(
                    PlacementResult.rejected(
                            RequestRegistry.buildErrorResponse(
                                    StrategyErrorType.NO_PREFILL_WORKER,
                                    null)));
            scheduler = new RequestScheduler(
                    configService,
                    router,
                    mock(EndpointRegistry.class),
                    mock(BatchSchedulerReporter.class),
                    evictionManager,
                    lifecycle);
        }
    }
}
