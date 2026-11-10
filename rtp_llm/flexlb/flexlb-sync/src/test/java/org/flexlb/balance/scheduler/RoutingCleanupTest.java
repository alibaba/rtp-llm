package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.junit.jupiter.api.Test;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies that failed resource cleanup ends the placement attempt and reports its failure.
 */
class RoutingCleanupTest {
    @Test
    void cleanupFailureFinishesAdmissionAsFailed() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.batchConfig(), 2L);
        var route = mock(RouteResult.class);
        var endpoint = mock(DecodeEndpoint.class);
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(2L), 3L);
        var failure = new IllegalStateException("cleanup failed");
        var handle = RequestProtocolTestSupport.admissionHandle(context);
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

        org.junit.jupiter.api.Assertions.assertSame(
                failure, AbstractRequestScheduler.finishUnpublishedRoute(route, handle));

        verify(RequestProtocolTestSupport.placementScheduler(handle))
                .completeRouting(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.eq(ContextResult.RoutingCompletion.FAILED),
                        any());
        verify(RequestProtocolTestSupport.placementScheduler(handle), never())
                .completeRouting(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.eq(ContextResult.RoutingCompletion.FINISHED),
                        org.mockito.ArgumentMatchers.isNull());
    }
}
