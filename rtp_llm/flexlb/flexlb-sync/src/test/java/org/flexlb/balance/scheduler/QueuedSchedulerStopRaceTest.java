package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.DecodePreemption;
import org.flexlb.balance.planner.BuiltinGroupPlanner;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.RejectedExecutionException;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class QueuedSchedulerStopRaceTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void rejectedExecutionAfterStopReleasesTheSlotWithoutRecordingARequestFailure(
            boolean stopping) {
        var config = SchedulingTestConfig.batchConfig();
        var lifecycle = mock(AbstractRequestScheduler.class);
        var requests = mock(RequestRepository.class);
        when(requests.snapshotActive()).thenReturn(java.util.List.of());
        ReflectionTestUtils.setField(lifecycle, "requests", requests);
        var tasks = mock(RequestTaskPool.class);
        ReflectionTestUtils.setField(lifecycle, "tasks", tasks);
        ReflectionTestUtils.setField(lifecycle, "dispatcher", mock(DeliveryDispatcher.class));
        ReflectionTestUtils.setField(lifecycle, "expirationTimer", mock(ExpirationTimer.class));
        ReflectionTestUtils.setField(
                lifecycle, "responseCompletions", mock(ResponseCompletionExecutor.class));
        ReflectionTestUtils.setField(
                lifecycle,
                "deliveryReporter",
                mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class));
        ReflectionTestUtils.setField(
                lifecycle, "cacheTrace", mock(org.flexlb.service.TheoryCacheHitReporter.class));
        var scheduler =
                SchedulerTestServices.queued(
                        config,
                        mock(RequestRouteSelector.class),
                        mock(DecodePreemption.class),
                        lifecycle,
                        new PlacementAvailability(),
                        new OrderedRequestQueue(false),
                        BuiltinGroupPlanner.from(config));
        var original = (ExecutorService) ReflectionTestUtils.getField(scheduler, "planners");
        var executor = mock(ExecutorService.class);
        var rejection =
                new RejectedExecutionException("pool stopped between selection and submission");
        when(executor.isShutdown()).thenReturn(stopping);
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
        try {
            scheduler.onEndpointChanged(prefill);
            assertDoesNotThrow(scheduler::schedule);
            assertEquals(0, ReflectionTestUtils.getField(scheduler, "queueSchedulingInFlight"));
            if (stopping) {
                verify(tasks, never()).recordFailure(rejection);
                verify(prefill, never()).closeQueue();
            } else {
                verify(tasks).recordFailure(rejection);
                verify(prefill).closeQueue();
            }
        } finally {
            ReflectionTestUtils.setField(scheduler, "planners", original);
            scheduler.close();
        }
    }
}
