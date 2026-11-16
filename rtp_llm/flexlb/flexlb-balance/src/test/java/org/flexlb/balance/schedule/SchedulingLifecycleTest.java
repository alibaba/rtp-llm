package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Keeps global resource shutdown out of the scheduler interface.
 */
class SchedulingLifecycleTest {
    @Test
    void schedulerDoesNotExposeGlobalShutdown() {
        assertThrows(
                NoSuchMethodException.class,
                () -> AbstractRequestScheduler.class.getMethod("shutdown", long.class));
    }

    @Test
    void localShutdownCancelsOnlyItsRequestsAndLeavesSharedResourcesOpen() {
        var requests = org.mockito.Mockito.mock(RequestRepository.class);
        var tasks = org.mockito.Mockito.mock(RequestTaskPool.class);
        var timer = org.mockito.Mockito.mock(ExpirationTimer.class);
        var dispatcher = org.mockito.Mockito.mock(EngineCancellation.class);
        var responses = org.mockito.Mockito.mock(ResponseCompletionExecutor.class);
        var endpoints = org.mockito.Mockito.mock(org.flexlb.worker.endpoint.EndpointRegistry.class);
        var scheduler =
                org.mockito.Mockito.spy(
                        new SchedulerTestServices.PassiveScheduler(
                                requests,
                                tasks,
                                timer,
                                dispatcher,
                                responses,
                                org.mockito.Mockito.mock(
                                        org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                                org.mockito.Mockito.mock(
                                        org.flexlb.balance.requestlog.TheoryCacheHitReporter
                                                .class)));
        var other = org.mockito.Mockito.mock(AbstractRequestScheduler.class);
        var ownRequest = org.mockito.Mockito.mock(RequestContext.class);
        var otherRequest = org.mockito.Mockito.mock(RequestContext.class);
        org.mockito.Mockito.when(ownRequest.getRequestScheduler()).thenReturn(scheduler);
        org.mockito.Mockito.when(otherRequest.getRequestScheduler()).thenReturn(other);
        org.mockito.Mockito.when(requests.snapshotActive(org.mockito.ArgumentMatchers.any()))
                .thenAnswer(
                        call -> {
                            java.util.function.Predicate<RequestContext> filter =
                                    call.getArgument(0);
                            return java.util.List.of(ownRequest, otherRequest).stream()
                                    .filter(filter)
                                    .toList();
                        });
        org.mockito.Mockito.doReturn(null)
                .when(scheduler)
                .cancel(ownRequest, CancelReason.SHUTDOWN);

        scheduler.stopAndCancelRequests();

        org.mockito.Mockito.verify(scheduler).close();
        org.mockito.Mockito.verify(scheduler).cancel(ownRequest, CancelReason.SHUTDOWN);
        org.mockito.Mockito.verify(scheduler, org.mockito.Mockito.never())
                .cancel(otherRequest, CancelReason.SHUTDOWN);
        org.mockito.Mockito.verify(requests, org.mockito.Mockito.never()).stopAccepting();
        org.mockito.Mockito.verify(requests, org.mockito.Mockito.never()).closeRegistration();
        org.mockito.Mockito.verifyNoInteractions(
                other, tasks, timer, dispatcher, responses, endpoints);
    }
}
