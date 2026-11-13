package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeStrategy;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.balance.strategy.VitStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that cancellation after route commit ends execution without replacing the selected response.
 */
class DirectRequestLifetimeRaceTest {
    @ParameterizedTest
    @EnumSource(CancelReason.class)
    void cancellationAfterRouteCommitKeepsSuccessResponse(CancelReason reason) throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var model = mock(ModelMetaConfig.class);
        when(model.requiredRoles()).thenReturn(List.of(RoleType.PDFUSION));
        var endpoints = mock(EndpointRegistry.class);
        var prefillStrategy =
                org.mockito.Mockito.spy(
                        new CostBasedPrefillStrategy(
                                service,
                                endpoints,
                                org.mockito.Mockito.mock(
                                        org.flexlb.cache.match.CacheAwareService.class),
                                org.mockito.Mockito.mock(
                                        org.flexlb.cache.metrics.CacheMetricsReporter.class)));
        var router =
                new RequestRouteSelector(
                        endpoints,
                        prefillStrategy,
                        mock(DecodeStrategy.class),
                        mock(VitStrategy.class),
                        org.mockito.Mockito.mock(EncoderStrategy.class),
                        model);
        var requests =
                org.flexlb.balance.schedule.SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class),
                        router,
                        mock(org.flexlb.balance.schedule.DecodePreemption.class),
                        new CapacityNotifier());
        try {
            var context = RequestProtocolTestSupport.context(config, 101L);
            var prefill = mock(PrefillEndpoint.class);
            when(prefill.acceptsNewRequests()).thenReturn(true);
            when(prefill.tryRegisterSelection(any(), any(), org.mockito.ArgumentMatchers.anyLong()))
                    .thenReturn(
                            new WorkerEndpoint.Registration<PrefillEndpoint.RequestRegistration>(
                                    WorkerEndpoint.RegistrationStatus.REGISTERED,
                                    mock(PrefillEndpoint.RequestRegistration.class)));
            var selectedEndpoint = prefill;
            var metadata = new ServerStatus();
            metadata.setSuccess(true);
            metadata.setRole(RoleType.PDFUSION);
            metadata.setRequestId(String.valueOf(101L));
            metadata.setServerIp("127.0.0.1");
            metadata.setHttpPort(8080);
            metadata.setGrpcPort(8081);
            when(selectedEndpoint.hasStoppedAcceptingRequests()).thenReturn(false);
            when(prefill.getIp()).thenReturn(metadata.getServerIp());
            when(prefill.getHttpPort()).thenReturn(metadata.getHttpPort());
            var reservation = mock(PrefillEndpoint.RequestRegistration.class);

            when(prefill.commitSelection(any())).thenReturn(true);
            var status =
                    org.flexlb.worker.status.WorkerStatus.createDiscovered(
                            RoleType.PDFUSION, null, "127.0.0.1", 8080, 8081, null);
            when(prefill.getStatus()).thenReturn(status);
            org.mockito.Mockito.doAnswer(
                            call ->
                                    SelectionTestSupport.selectionResult(
                                            call,
                                            RoutingResult.success(
                                                    new org.flexlb.balance.strategy
                                                            .CostBasedPrefillStrategy.Selection(
                                                            "127.0.0.1:8080",
                                                            status.getGenerationId(),
                                                            30_000L,
                                                            0L,
                                                            0L,
                                                            0L,
                                                            null,
                                                            prefill,
                                                            null,
                                                            prefill.selectionSnapshot()))))
                    .when(prefillStrategy)
                    .chooseBest(
                            org.mockito.ArgumentMatchers.eq(context.getRequest()),
                            org.mockito.ArgumentMatchers.anyList(),
                            org.mockito.ArgumentMatchers.any());
            var state = prefill;
            org.flexlb.worker.endpoint.EndpointTestSupport.stubResources(prefill, state);
            when(endpoints.<org.flexlb.worker.endpoint.PrefillEndpoint>endpoints(
                            org.mockito.ArgumentMatchers.eq(RoleType.PDFUSION),
                            org.mockito.ArgumentMatchers.any()))
                    .thenReturn(java.util.List.of(prefill));
            when(endpoints.getEndpoint(RoleType.PDFUSION, "127.0.0.1:8080"))
                    .thenReturn(selectedEndpoint);

            AtomicBoolean cleanupProcessed = new AtomicBoolean();
            AtomicBoolean cancellationInjected = new AtomicBoolean();
            doAnswer(
                            invocation -> {
                                if (!cancellationInjected.compareAndSet(false, true)) {
                                    return invocation.callRealMethod();
                                }
                                RequestContext requestContext =
                                        requests.repository.findActive("101");
                                if (reason != CancelReason.DEADLINE_EXCEEDED) {
                                    requests.cancel("101", reason);
                                }
                                RequestProtocolTestSupport.expireInactiveRequest(
                                        requests,
                                        requestContext,
                                        requestContext.snapshot().inactivityExpiresAtMs());
                                assertTrue(
                                        org.flexlb.balance.schedule.SchedulerTestSupport.repository(
                                                        requests)
                                                .getRequestState("101")
                                                .state()
                                                .isTerminal());
                                Object result = invocation.callRealMethod();
                                cleanupProcessed.set(true);
                                return result;
                            })
                    .when(requests)
                    .processPendingRequestActions(
                            org.mockito.ArgumentMatchers.isA(RequestContext.class));

            var returned =
                    SchedulerTestServices.entry(
                                    (AbstractRequestScheduler)
                                            org.flexlb.balance.schedule.SchedulerTestServices
                                                    .scheduler(requests))
                            .submit(context);
            assertSame(context.getFuture(), returned);
            var response = returned.get(2L, TimeUnit.SECONDS);
            SchedulerTestSupport.awaitRequestWork(SchedulerTestSupport.runtime(requests));
            assertFalse(returned.isCompletedExceptionally());
            assertTrue(response.isSuccess());
            assertTrue(
                    cleanupProcessed.get(),
                    "cancellation must finish execution without replacing the response");
            verify(prefill)
                    .commitSelection(
                            org.mockito.ArgumentMatchers.argThat(route -> route.ctx() == context));
            org.flexlb.worker.endpoint.EndpointTestSupport.closeAsync(
                    verify(selectedEndpoint, org.mockito.Mockito.never()));
        } finally {
            RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(requests);
            requests.closeOutstandingAndTerminalize();
            SchedulerTestSupport.timer(
                            org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requests))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requests));
        }
    }
}
