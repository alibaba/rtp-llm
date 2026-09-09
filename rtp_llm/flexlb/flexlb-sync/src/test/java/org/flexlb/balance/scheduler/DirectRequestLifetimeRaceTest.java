package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DirectRequestLifetimeRaceTest {
    @ParameterizedTest
    @EnumSource(CancelReason.class)
    void localExpiryBeforeRouteConfirmationKeepsCanonicalCancellationResponse(CancelReason reason)
            throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(SchedulerConfig.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var requests = spy(new RequestRegistry(service, mock(BatchSchedulerReporter.class),
                mock(RequestSchedulerReporter.class)));
        try {
            var context = RequestLifecycleTestSupport.context(config, 101L);
            var prefill = mock(PrefillEndpoint.class);
            var pin = mock(WorkerEndpoint.GenerationPin.class);
            when(pin.endpoint()).thenReturn(prefill);
            var metadata = new ServerStatus();
            metadata.setSuccess(true);
            metadata.setRole(RoleType.PDFUSION);
            metadata.setRequestId(101L);
            metadata.setServerIp("127.0.0.1");
            metadata.setHttpPort(8080);
            metadata.setGrpcPort(8081);
            var selection = mock(SelectedRole.class);
            when(selection.serverStatus()).thenReturn(metadata);
            when(selection.prefillWorkMs()).thenReturn(30_000L);
            when(selection.takeGenerationPin()).thenReturn(pin);
            var reservation = mock(PrefillState.RouteReservation.class);
            when(prefill.reserveUnqueuedRoute(eq(pin), any(), eq(30_000L)))
                    .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, reservation));
            var routeCommit = mock(PrefillEndpoint.RouteCommitAdmission.class);
            when(prefill.tryBeginRouteCommitAdmission()).thenReturn(routeCommit);
            var handoff = mock(PrefillState.CommittedHandoff.class);
            when(handoff.precedingWork()).thenReturn(new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L));
            when(routeCommit.commit(any(), any())).thenReturn(handoff);
            var prefillSelector = mock(CostBasedPrefillStrategy.class);
            when(prefillSelector.select(context, RoleType.PDFUSION, null))
                    .thenReturn(PlacementResult.success(selection));
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
            var router = new DefaultRouter(prefillSelector, mock(DecodeSelector.class),
                    mock(RandomStrategy.class), service, model);

            doAnswer(invocation -> {
                invocation.callRealMethod();
                RequestSlot slot = requests.requestSlot(101L);
                if (reason == CancelReason.CLIENT_CANCELLED) {
                    requests.cancelRequest(101L, 0L, reason);
                }
                requests.expireInactiveRequest(slot, slot.createdAtMs()
                        + config.getRequestLifecycle().getRequest().getTimeoutMs());
                assertTrue(requests.getRequestState(101L, 0L).state().isTerminal());
                return null;
            }).when(requests).beginDelivery(any(), any(), org.mockito.ArgumentMatchers.anyLong());
            AtomicBoolean lateConfirmation = new AtomicBoolean();
            doAnswer(invocation -> {
                assertTrue(requests.getRequestState(101L, 0L).state().isTerminal());
                Object result = invocation.callRealMethod();
                lateConfirmation.set(true);
                return result;
            }).when(requests).complete(any(), any());

            var scheduler = new RequestScheduler(service, router,
                    mock(org.flexlb.balance.endpoint.EndpointRegistry.class), mock(BatchSchedulerReporter.class),
                    mock(org.flexlb.balance.eviction.EvictionManager.class), requests, new PlacementAvailability());
            var returned = scheduler.submit(context);
            assertSame(context.getFuture(), returned);
            var response = returned.get(2L, TimeUnit.SECONDS);
            assertFalse(returned.isCompletedExceptionally());
            assertFalse(response.isSuccess());
            assertEquals((reason == CancelReason.DEADLINE_EXCEEDED
                    ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.REQUEST_CANCELLED).getErrorCode(),
                    response.getCode());
            assertTrue(lateConfirmation.get(), "confirmation after cancellation must return without reopening the request");
            verify(routeCommit).commit(any(), eq(List.of(reservation)));
            verify(pin).close();
        } finally {
            requests.closeAdmissionAndAwaitMutations();
            requests.closeOutstandingAndTerminalize();
            requests.closeExpiration();
            requests.closePublisher();
        }
    }
}
