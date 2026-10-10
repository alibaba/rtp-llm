package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigTestSupport;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
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
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DirectRequestLifetimeRaceTest {
    @ParameterizedTest
    @EnumSource(CancelReason.class)
    void localExpiryBeforeRouteConfirmationKeepsCanonicalCancellationResponse(CancelReason reason)
            throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        SchedulingTestConfig.useNonBatchDispatcher(config);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var requests = org.flexlb.balance.scheduler.SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                mock(RecentCacheKeyTraceReporter.class));
        try {
            var context = RequestProtocolTestSupport.context(config, 101L);
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
            var selection = mock(WorkerAssignment.class);
            when(selection.serverStatus()).thenReturn(metadata);
            when(selection.requestId()).thenReturn(metadata.getRequestId());
            when(selection.role()).thenReturn(metadata.getRole());
            when(selection.group()).thenReturn(metadata.getGroup());
            when(selection.prefillWorkMs()).thenReturn(30_000L);
            when(selection.generationPin()).thenReturn(pin);
            when(selection.endpoint()).thenReturn(prefill);
            var selectionOpen = new java.util.concurrent.atomic.AtomicBoolean(true);
            org.mockito.Mockito.doAnswer(call -> {
                if (selectionOpen.compareAndSet(true, false)) { pin.close(); }
                return null;
            }).when(selection).close();
            var reservation = mock(PrefillState.RouteReservation.class);
            when(prefill.reserveUnqueuedRoute(eq(pin), any(), eq(30_000L)))
                    .thenReturn(new PrefillState.ReservationResult<>(PrefillState.CapacityStatus.ACQUIRED, reservation));
            var handoff = mock(PrefillState.CommittedHandoff.class);
            var capture = mock(PrefillState.WorkCapture.class);
            when(capture.materialize()).thenReturn(new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L));
            when(handoff.precedingWork()).thenReturn(capture);
            when(prefill.commit(any(PrefillState.CommitPlan.class))).thenReturn(handoff);
            var prefillSelector = mock(CostBasedPrefillStrategy.class);
            when(prefillSelector.select(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PDFUSION, null))
                    .thenReturn(PlacementResult.success(selection));
            var model = mock(ModelMetaConfig.class);
            when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
            var router = new RequestWorkerSelector(prefillSelector, mock(DecodeSelector.class),
                    mock(VitWorkerSelector.class), model);

            AtomicBoolean lateConfirmation = new AtomicBoolean();
            doAnswer(invocation -> {
                RequestContext requestContext = requests.requests.findActive(101L);
                if (reason != CancelReason.DEADLINE_EXCEEDED) { requests.cancel(101L, 0L, reason); }
                RequestProtocolTestSupport.expireInactiveRequest(requests, requestContext, requestContext.snapshot().inactivityExpiresAtMs());
                assertTrue(org.flexlb.balance.scheduler.SchedulerTestSupport.repository(requests).getRequestState(101L, 0L).state().isTerminal());
                Object result = invocation.callRealMethod();
                lateConfirmation.set(true);
                return result;
            }).when(requests).onDeliveryEvidence(any(), org.mockito.ArgumentMatchers.isA(ContextResult.RouteDelivered.class));

            var scheduler = org.flexlb.balance.scheduler.SchedulerTestSupport.configure(requests, service.loadBalanceConfig(), router, mock(DeliveryMetricsReporter.class), mock(org.flexlb.balance.eviction.DecodeCapacityAcquirer.class), new PlacementAvailability());
            var returned = scheduler.submit(context);
            assertSame(context.data().getFuture(), returned);
            var response = returned.get(2L, TimeUnit.SECONDS);
            assertFalse(returned.isCompletedExceptionally());
            assertFalse(response.isSuccess());
            assertEquals((reason == CancelReason.DEADLINE_EXCEEDED
                    ? StrategyErrorType.BATCH_SLO_EXPIRED : StrategyErrorType.REQUEST_CANCELLED).getErrorCode(),
                    response.getCode());
            assertTrue(lateConfirmation.get(), "confirmation after cancellation must return without reopening the request");
            verify(prefill).commit(org.mockito.ArgumentMatchers.argThat(plan ->
                    plan.routeReservations().equals(List.of(reservation)) && plan.batchReservation() == null
                            && plan.predictions().length == 0 && plan.items().getFirst().ctx() == context));
            verify(pin).close();
        } finally {
            RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(requests);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests)).close();
            SchedulerTestSupport.closeRequestExecutors(org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests));
        }
    }
}
