package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
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
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that cancellation before direct-route confirmation keeps its original response and cleanup ownership.
 */
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
        var prefillSelector =
                mock(
                        org.flexlb.balance.strategy.PrefillSelectionStrategy.class,
                        org.mockito.Mockito.CALLS_REAL_METHODS);
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
        var endpoints = mock(org.flexlb.balance.endpoint.EndpointRegistry.class);
        var router =
                new RequestRouteSelector(
                        endpoints,
                        prefillSelector,
                        mock(DecodeSelector.class),
                        mock(VitWorkerSelector.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.strategy.EncoderSelectionStrategy.class),
                        org.mockito.Mockito.mock(org.flexlb.cache.match.CacheAwareService.class),
                        null,
                        mock(org.flexlb.service.monitor.EngineHealthReporter.class),
                        mock(org.flexlb.cache.telemetry.CacheMetricsReporter.class),
                        service,
                        model);
        var requests =
                org.flexlb.balance.scheduler.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class),
                        router,
                        mock(org.flexlb.balance.preemption.DecodePreemption.class),
                        new CapacityNotifier());
        try {
            var context = RequestProtocolTestSupport.context(config, 101L);
            var prefill = mock(PrefillEndpoint.class);
            when(prefill.acceptsNewRequests()).thenReturn(true);
            when(prefill.tryRegisterSelection(
                            any(),
                            any(),
                            org.mockito.ArgumentMatchers.anyLong(),
                            org.mockito.ArgumentMatchers.anyBoolean()))
                    .thenReturn(
                            new PrefillEndpoint.Registration(
                                    RoutingResult.Registration.REGISTERED,
                                    mock(PrefillEndpoint.RequestRecord.class)));
            var selectedEndpoint = prefill;
            var metadata = new ServerStatus();
            metadata.setSuccess(true);
            metadata.setRole(RoleType.PDFUSION);
            metadata.setRequestId(String.valueOf(101L));
            metadata.setServerIp("127.0.0.1");
            metadata.setHttpPort(8080);
            metadata.setGrpcPort(8081);
            when(selectedEndpoint.isRetired()).thenReturn(false);
            when(prefill.getIp()).thenReturn(metadata.getServerIp());
            when(prefill.getHttpPort()).thenReturn(metadata.getHttpPort());
            var reservation = mock(PrefillEndpoint.RequestRecord.class);

            var capture =
                    mock(org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot.class);
            when(capture.calculate())
                    .thenReturn(
                            new WorkSnapshot(
                                    System.currentTimeMillis(),
                                    java.util.List.of(),
                                    java.util.List.of(),
                                    0L));
            when(prefill.commitDirect(any()))
                    .thenAnswer(
                            call -> {
                                RouteResult route = call.getArgument(0);

                                return capture;
                            });
            var status =
                    org.flexlb.dao.master.WorkerStatus.createDiscovered(
                            RoleType.PDFUSION, null, "127.0.0.1", 8080, 8081, null);
            when(prefill.getStatus()).thenReturn(status);
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.registerResult(
                                            call,
                                            RoutingResult.success(
                                                    new org.flexlb.balance.strategy
                                                            .PrefillSelectionStrategy.Selection<>(
                                                            "127.0.0.1:8080",
                                                            status.getGenerationId(),
                                                            0L,
                                                            30_000L,
                                                            30_000L,
                                                            0L,
                                                            0L,
                                                            0L,
                                                            null,
                                                            1,
                                                            new EndpointRegistry
                                                                    .PrefillRoutingEntry(
                                                                    "127.0.0.1:8080", prefill),
                                                            null,
                                                            new EndpointRegistry
                                                                            .PrefillRoutingEntry(
                                                                            "127.0.0.1:8080",
                                                                            prefill)
                                                                    .projectionInputs()))))
                    .when(prefillSelector)
                    .selectAndRegister(
                            org.mockito.ArgumentMatchers.eq(context.data().getRequest()),
                            org.mockito.ArgumentMatchers.eq(context.data().getSchedulingMetadata()),
                            org.mockito.ArgumentMatchers.eq(RoleType.PDFUSION),
                            org.mockito.ArgumentMatchers.eq(null),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any(java.util.function.LongSupplier.class),
                            org.mockito.ArgumentMatchers.any());
            var state = prefill;
            when(state.requestCapacityAvailable(
                            org.mockito.ArgumentMatchers.anyLong(),
                            org.mockito.ArgumentMatchers.anyInt()))
                    .thenReturn(true);
            org.flexlb.balance.endpoint.EndpointTestSupport.stubResources(prefill, state);
            when(endpoints.prefillRoutingSnapshot(RoleType.PDFUSION))
                    .thenReturn(
                            java.util.List.of(
                                    new org.flexlb.balance.endpoint.EndpointRegistry
                                            .PrefillRoutingEntry("127.0.0.1:8080", prefill)));
            when(endpoints.selectEndpoint(RoleType.PDFUSION, "127.0.0.1:8080"))
                    .thenReturn(selectedEndpoint);

            AtomicBoolean lateConfirmation = new AtomicBoolean();
            doAnswer(
                            invocation -> {
                                RequestContext requestContext =
                                        requests.repository.findActive("101");
                                if (reason != CancelReason.DEADLINE_EXCEEDED) {
                                    requests.cancel("101", 0L, reason);
                                }
                                RequestProtocolTestSupport.expireInactiveRequest(
                                        requests,
                                        requestContext,
                                        requestContext.snapshot().inactivityExpiresAtMs());
                                assertTrue(
                                        org.flexlb.balance.scheduler.SchedulerTestSupport
                                                .repository(requests)
                                                .getRequestState("101", 0L)
                                                .state()
                                                .isTerminal());
                                Object result = invocation.callRealMethod();
                                lateConfirmation.set(true);
                                return result;
                            })
                    .when(requests)
                    .onDeliveryResult(
                            org.mockito.ArgumentMatchers.isA(ContextResult.RouteDelivered.class));

            var returned =
                    SchedulerTestServices.entry(
                                    (AbstractRequestScheduler)
                                            org.flexlb.balance.scheduler.SchedulerTestServices
                                                    .scheduler(requests))
                            .submit(context);
            assertSame(context.data().getFuture(), returned);
            var response = returned.get(2L, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(requests));
            assertFalse(returned.isCompletedExceptionally());
            assertFalse(response.isSuccess());
            assertEquals(
                    (reason == CancelReason.DEADLINE_EXCEEDED
                                    ? StrategyErrorType.BATCH_SLO_EXPIRED
                                    : StrategyErrorType.REQUEST_CANCELLED)
                            .getErrorCode(),
                    response.getCode());
            assertTrue(
                    lateConfirmation.get(),
                    "confirmation after cancellation must return without reopening the request");
            verify(prefill)
                    .commitDirect(
                            org.mockito.ArgumentMatchers.argThat(route -> route.ctx() == context));
            verify(prefill, org.mockito.Mockito.never()).tryStartQueuedDelivery(any());
            verify(selectedEndpoint, org.mockito.Mockito.never()).close();
        } finally {
            RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(requests);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(requests));
        }
    }
}
