package org.flexlb.balance.schedule;

import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.PrefillEndpoint;
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
    void selectionCompletionFailureIsReturnedToTheCaller() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 3L);
        var handle = RequestProtocolTestSupport.admissionHandle(context);
        var failure = new IllegalStateException("completion failed");
        doThrow(failure)
                .when(RequestProtocolTestSupport.placementScheduler(handle))
                .finishSelectionWithoutCommit(any(), any());

        org.junit.jupiter.api.Assertions.assertSame(
                failure, AbstractRequestScheduler.finishUnpublishedRoute(null, handle));
    }

    @Test
    void cleanupFailureFinishesAdmissionAsFailed() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 2L);
        var route = mock(RequestResult.class);
        var endpoint = mock(DecodeEndpoint.class);
        var reservation = new DecodeEndpoint.RequestRegistration(1L, String.valueOf(2L));
        var failure = new IllegalStateException("cleanup failed");
        var handle = RequestProtocolTestSupport.admissionHandle(context);
        when(route.prefillEndpoint()).thenReturn(mock(PrefillEndpoint.class));
        when(route.decodeEndpoint()).thenReturn(endpoint);
        when(route.ctx()).thenReturn(context);
        {
            when(route.decodeRegistration()).thenReturn(reservation);
            when(endpoint.registrationState(reservation))
                    .thenReturn(DecodeResources.RegistrationState.UNCONFIRMED);
            doThrow(failure).when(endpoint).rollbackSelection(reservation);
        }

        org.junit.jupiter.api.Assertions.assertSame(
                failure, AbstractRequestScheduler.finishUnpublishedRoute(route, handle));

        verify(RequestProtocolTestSupport.placementScheduler(handle))
                .finishSelectionWithoutCommit(org.mockito.ArgumentMatchers.same(handle), any());
        verify(RequestProtocolTestSupport.placementScheduler(handle), never())
                .finishSelectionWithoutCommit(
                        org.mockito.ArgumentMatchers.same(handle),
                        org.mockito.ArgumentMatchers.isNull());
    }
}
