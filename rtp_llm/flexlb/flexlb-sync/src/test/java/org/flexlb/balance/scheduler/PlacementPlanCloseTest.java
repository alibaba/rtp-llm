package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PlacementPlanCloseTest {
    @Test
    void cleanupFailureStillFinishesAdmissionOnce() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 2L);
        var route = mock(RouteExecution.class);
        var endpoint = mock(DecodeEndpoint.class);
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(2L), 3L);
        var failure = new IllegalStateException("cleanup failed");
        var plan = RequestProtocolTestSupport.plan(context, route);
        var handle = RequestProtocolTestSupport.admissionHandle(context);
        ReflectionTestUtils.setField(plan, "handle", handle);
        doAnswer(
                        call -> {
                            assertFalse(
                                    Thread.holdsLock(plan),
                                    "lifecycle callbacks must run outside the Plan lock");
                            return null;
                        })
                .when(RequestProtocolTestSupport.placementScheduler(handle))
                .completePlacement(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.eq(ContextResult.PublicationResult.PUBLISHED),
                        org.mockito.ArgumentMatchers.isNull());
        doAnswer(
                        call -> {
                            assertFalse(
                                    Thread.holdsLock(plan),
                                    "lifecycle callbacks must run outside the Plan lock");
                            return null;
                        })
                .when(RequestProtocolTestSupport.placementScheduler(handle))
                .completePlacement(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.eq(ContextResult.PublicationResult.FAILED),
                        any());
        when(route.prefillEndpoint())
                .thenReturn(mock(org.flexlb.balance.endpoint.PrefillEndpoint.class));
        when(route.decodeEndpoint()).thenReturn(endpoint);
        when(route.ctx()).thenReturn(context);
        {
            when(route.decodeReservation()).thenReturn(reservation);
            doThrow(failure)
                    .when(endpoint)
                    .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        }

        var runtime = mock(RequestTaskPool.class);
        var scheduler = ReflectionTestUtils.getField(plan, "this$0");
        var lifecycle = scheduler;
        ReflectionTestUtils.setField(lifecycle, "tasks", runtime);
        plan.close();
        plan.close();
        verify(runtime).recordFailure(failure);

        verify(RequestProtocolTestSupport.placementScheduler(handle))
                .completePlacement(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.eq(ContextResult.PublicationResult.FAILED),
                        any());
        verify(RequestProtocolTestSupport.placementScheduler(handle), never())
                .completePlacement(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.eq(ContextResult.PublicationResult.PUBLISHED),
                        org.mockito.ArgumentMatchers.isNull());
    }
}
