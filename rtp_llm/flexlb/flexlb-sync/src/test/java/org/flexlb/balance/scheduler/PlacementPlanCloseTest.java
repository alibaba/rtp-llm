package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.*;

class PlacementPlanCloseTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cleanupFailureStillFinishesAdmissionOnce(boolean rollbackFails) {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 2L);
        var route = mock(RequestRoute.class);
        var endpoint = mock(DecodeEndpoint.class);
        var reservation = new DecodeResources.ReservationHandle(1L, 2L, 3L);
        var failure = new IllegalStateException("cleanup failed");
        var plan = RequestProtocolTestSupport.plan(context, route);
        var handle = mock(AdmissionHandle.class);
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
        when(route.decodeEp()).thenReturn(endpoint);
        if (rollbackFails) {
            ReflectionTestUtils.setField(plan, "pendingDecodeRollback", reservation);
            doThrow(failure)
                    .when(endpoint)
                    .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
        } else {
            doThrow(failure).when(route).close();
        }

        var runtime = mock(RequestTaskPool.class);
        plan.close(runtime);
        plan.close(runtime);
        verify(runtime).recordFailure(failure);

        var order = inOrder(route, handle);
        order.verify(route).close();
        if (rollbackFails) {
            order.verify(handle).terminate(any());
            verify(handle, never()).finish();
        } else {
            order.verify(handle).finish();
            verify(handle, never()).terminate(any());
        }
        verifyNoMoreInteractions(handle);
    }
}
