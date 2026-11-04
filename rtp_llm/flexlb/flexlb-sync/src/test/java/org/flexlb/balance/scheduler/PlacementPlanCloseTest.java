package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.inOrder;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoMoreInteractions;
import static org.mockito.Mockito.when;

class PlacementPlanCloseTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cleanupFailureStillFinishesAdmissionOnce(boolean rollbackFails) {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 2L);
        var route = mock(RequestRoute.class);
        var endpoint = mock(DecodeEndpoint.class);
        var reservation = new DecodeResources.ReservationHandle(1L, String.valueOf(2L), 3L);
        var failure = new IllegalStateException("cleanup failed");
        var plan = RequestProtocolTestSupport.plan(context, route);
        var handle = mock(AdmissionHandle.class);
        when(handle.owner()).thenReturn(context);
        ReflectionTestUtils.setField(plan, "handle", handle);
        doAnswer(
                        call -> {
                            assertFalse(
                                    Thread.holdsLock(plan),
                                    "lifecycle callbacks must run outside the Plan lock");
                            return null;
                        })
                .when(handle)
                .finish();
        doAnswer(
                        call -> {
                            assertFalse(
                                    Thread.holdsLock(plan),
                                    "lifecycle callbacks must run outside the Plan lock");
                            return null;
                        })
                .when(handle)
                .terminate(any());
        when(route.prefillEp()).thenReturn(mock(org.flexlb.balance.endpoint.PrefillEndpoint.class));
        when(route.decodeEp()).thenReturn(endpoint);
        when(route.ctx()).thenReturn(context);
        if (rollbackFails) {
            when(route.decodeReservation()).thenReturn(reservation);
            doThrow(failure)
                    .when(endpoint)
                    .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        } else {
            doThrow(failure).when(route).close();
        }

        var runtime = mock(RequestTaskPool.class);
        var scheduler = ReflectionTestUtils.getField(plan, "this$0");
        var lifecycle = ReflectionTestUtils.getField(scheduler, "lifecycle");
        ReflectionTestUtils.setField(lifecycle, "tasks", runtime);
        plan.close();
        plan.close();
        verify(runtime).recordFailure(failure);

        var order = inOrder(route, handle);
        order.verify(route).close();
        if (rollbackFails) {
            order.verify(handle).terminate(any());
            verify(handle, never()).finish();
        } else {
            verify(handle).owner();
            order.verify(handle).finish();
            verify(handle, never()).terminate(any());
        }
        verifyNoMoreInteractions(handle);
    }
}
