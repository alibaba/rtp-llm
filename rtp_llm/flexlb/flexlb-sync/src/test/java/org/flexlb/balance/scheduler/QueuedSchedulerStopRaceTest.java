package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.balance.preemption.DecodePreemption;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks that executor rejection after scheduler stop returns the planning slot without failing the request twice.
 */
class QueuedSchedulerStopRaceTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void rejectedExecutionAfterStopReleasesTheSlotWithoutRecordingARequestFailure(boolean stopping)
            throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        var requests = mock(RequestRepository.class);
        when(requests.snapshotActive()).thenReturn(java.util.List.of());
        var tasks = mock(RequestTaskPool.class);
        var scheduler =
                new QueuedRequestScheduler(
                        config,
                        mock(RequestRouteSelector.class),
                        mock(DecodePreemption.class),
                        requests,
                        tasks,
                        mock(ExpirationTimer.class),
                        mock(DeliveryCoordinator.class),
                        mock(ResponseCompletionExecutor.class),
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.TheoryCacheHitReporter.class),
                        new CapacityNotifier(),
                        new OrderedRequestQueue(false),
                        BuiltinGroupPlanner.from(config),
                        mock(org.flexlb.balance.endpoint.EndpointRegistry.class));
        var original = (ExecutorService) ReflectionTestUtils.getField(scheduler, "planners");
        var executor = mock(ExecutorService.class);
        var rejection =
                new RejectedExecutionException("pool stopped between selection and submission");
        when(executor.isShutdown()).thenReturn(stopping);
        when(executor.isTerminated()).thenReturn(true);
        doAnswer(
                        call -> {
                            if (stopping) scheduler.stop();
                            throw rejection;
                        })
                .when(executor)
                .execute(any(Runnable.class));
        var prefill = mock(PrefillEndpoint.class, RETURNS_DEEP_STUBS);
        org.flexlb.balance.endpoint.EndpointTestSupport.useStateLock(
                prefill, new java.util.concurrent.locks.ReentrantLock());
        when(prefill.queueEnabled()).thenReturn(true);
        when(prefill.ipPort()).thenReturn("127.0.0.1:1");
        when(prefill.getStatus().getRole()).thenReturn(RoleType.PREFILL);
        ReflectionTestUtils.setField(scheduler, "planners", executor);
        var prefillDelivery = ReflectionTestUtils.getField(scheduler, "prefillDelivery");
        ReflectionTestUtils.setField(prefillDelivery, "planners", executor);
        try {
            scheduler.onEndpointChanged(prefill);
            scheduler.start();
            RequestProtocolTestSupport.awaitCondition(
                    () ->
                            stopping
                                    ? scheduler.awaitStopped(System.nanoTime())
                                    : ReflectionTestUtils.getField(
                                                    scheduler, "requestProcessingFailure")
                                            != null);
            scheduler.stop();
            org.junit.jupiter.api.Assertions.assertTrue(
                    scheduler.awaitStopped(
                            System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(2)));
            assertEquals(
                    0, ReflectionTestUtils.getField(prefillDelivery, "queueSchedulingInFlight"));
            if (stopping) {
                org.junit.jupiter.api.Assertions.assertNull(
                        ReflectionTestUtils.getField(scheduler, "requestProcessingFailure"));
                verify(prefill, never()).closeQueue();
            } else {
                org.junit.jupiter.api.Assertions.assertSame(
                        rejection,
                        ReflectionTestUtils.getField(scheduler, "requestProcessingFailure"));
                verify(prefill).closeQueue();
            }
        } finally {
            scheduler.stop();
            scheduler.awaitStopped(
                    System.nanoTime() + java.util.concurrent.TimeUnit.SECONDS.toNanos(2));
            ReflectionTestUtils.setField(scheduler, "planners", original);
            scheduler.close();
        }
    }
}
