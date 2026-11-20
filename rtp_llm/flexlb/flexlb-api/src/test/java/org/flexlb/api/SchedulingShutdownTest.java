package org.flexlb.api;

import org.flexlb.api.metrics.ApiMetricsReporter;
import org.flexlb.api.server.FlexlbGrpcServer;
import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.request.Request;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.AbstractRequestScheduler;
import org.flexlb.balance.schedule.EngineRequestCanceller;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestSchedulerTestRuntime;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.SelectionResult;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.master.MasterStatusService;
import org.flexlb.util.ApplicationWarmupState;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.mock.env.MockEnvironment;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

/**
 * The application drains ingress before stopping scheduling and shared resources.
 */
class SchedulingShutdownTest {
    @Test
    void drainsIngressBeforeClosingSchedulingOnce() {
        var scheduling = mock(AbstractRequestScheduler.class);
        var grpc = mock(FlexlbGrpcServer.class);
        var application = application(scheduling, grpc);
        application.offline();
        application.offline();
        var order = inOrder(grpc, scheduling);
        order.verify(scheduling).checkShutdownThread();
        order.verify(grpc).drain();
        order.verify(scheduling).stopAndCancelRequests();
        verify(scheduling).stopAndCancelRequests();
    }

    @Test
    void reportsSchedulingShutdownFailure() {
        var scheduling = mock(AbstractRequestScheduler.class);
        var failure = new IllegalStateException("scheduling cleanup failed");
        doThrow(failure).when(scheduling).stopAndCancelRequests();
        assertSame(
                failure,
                assertThrows(
                        IllegalStateException.class,
                        () -> application(scheduling, mock(FlexlbGrpcServer.class)).offline()));
    }

    @Test
    void rejectsUnsafeShutdownBeforeWaitingForApplicationMonitor() throws Exception {
        var scheduling = mock(AbstractRequestScheduler.class);
        var grpc = mock(FlexlbGrpcServer.class);
        doThrow(new IllegalStateException("owned callback")).when(scheduling).checkShutdownThread();
        var application = application(scheduling, grpc);
        synchronized (application) {
            java.util.concurrent.CompletableFuture.runAsync(
                            () -> assertThrows(IllegalStateException.class, application::offline))
                    .get(2, java.util.concurrent.TimeUnit.SECONDS);
        }
        verifyNoInteractions(grpc);
    }

    @ParameterizedTest(name = "queued={0}")
    @ValueSource(booleans = {false, true})
    void rejectsShutdownFromSelectionBeforeWaitingForApplicationMonitor(boolean queued)
            throws Exception {
        var config = new FlexlbConfig();
        if (!queued) config.setScheduler(ConfigTestSupport.direct());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var engine = mock(EngineGrpcClient.class);
        var grpc = mock(FlexlbGrpcServer.class);
        // If the guard misses, allow the callback to exit once the test releases the monitor.
        doThrow(new IllegalStateException("test stops before draining")).when(grpc).drain();

        try (var runtime =
                new RequestSchedulerTestRuntime(
                        service,
                        new EngineRequestCanceller(engine),
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        engine)) {
            var application = application(runtime.scheduler(), grpc);
            var router = mock(RequestRouteSelector.class);
            var enteringOffline = new CountDownLatch(1);
            var shutdownOutcome = new CompletableFuture<Throwable>();
            when(router.selectAndRegister(any(), any()))
                    .thenAnswer(
                            call -> {
                                enteringOffline.countDown();
                                try {
                                    application.offline();
                                    shutdownOutcome.complete(null);
                                } catch (RuntimeException failure) {
                                    shutdownOutcome.complete(failure);
                                }
                                return SelectionResult.rejected(
                                        Response.error(StrategyErrorType.NO_PREFILL_WORKER));
                            });
            runtime.bindRouter(router);
            var request = new RequestContext(config);
            request.setRequest(Request.builder().requestId("101").seqLen(16).build());
            request.setSchedulingMetadata(SchedulingMetadata.of(50, null, Long.MAX_VALUE, 50));

            synchronized (application) {
                runtime.entry().submit(request);
                assertTrue(
                        enteringOffline.await(3, TimeUnit.SECONDS), "Selection must call offline");
                var failure =
                        assertDoesNotThrow(
                                () -> shutdownOutcome.get(2, TimeUnit.SECONDS),
                                "offline must reject the selection thread while the monitor is held");
                assertInstanceOf(IllegalStateException.class, failure);
            }
            verifyNoInteractions(grpc);
        }
    }

    private static ApplicationLifecycle application(
            AbstractRequestScheduler scheduling, FlexlbGrpcServer grpc) {
        return new ApplicationLifecycle(
                mock(MasterStatusService.class),
                grpc,
                mock(ApiMetricsReporter.class),
                new MockEnvironment(),
                new ApplicationWarmupState(),
                scheduling,
                mock(org.flexlb.balance.schedule.RequestRepository.class),
                mock(java.util.concurrent.ExecutorService.class),
                mock(org.flexlb.balance.schedule.ExpirationTimer.class),
                mock(org.flexlb.balance.schedule.EngineRequestCanceller.class),
                mock(org.flexlb.balance.schedule.ResponseCompletionExecutor.class),
                mock(org.flexlb.worker.endpoint.EndpointRegistry.class),
                0L);
    }
}
