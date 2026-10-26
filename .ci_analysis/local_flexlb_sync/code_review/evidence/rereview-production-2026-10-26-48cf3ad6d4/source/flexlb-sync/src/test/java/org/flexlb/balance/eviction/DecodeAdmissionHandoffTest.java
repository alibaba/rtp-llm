package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.scheduler.QueuedRequestScheduler;
import org.flexlb.balance.scheduler.RequestContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class DecodeAdmissionHandoffTest {
    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void abandonedObserverCannotSkipReservationHandoff(boolean cancel, boolean accepted) {
        var admission = DecodeAdmissionTestSupport.mock();
        var plan = mock(QueuedRequestScheduler.Plan.class);
        var context = mock(RequestContext.class);
        var endpoint = mock(DecodeEndpoint.class);
        var reservation = new DecodeResources.ReservationHandle(1L, 42L, 1L);
        var source = new CompletableFuture<DecodeAdmission.PreemptionResult>();
        when(plan.context()).thenReturn(context);
        when(plan.accept(endpoint, reservation)).thenReturn(accepted);
        when(admission.tryReclaim(context, endpoint)).thenReturn(source);
        when(endpoint.release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK))
                .thenReturn(
                        DecodeResources.ResourceChange.of(
                                DecodeResources.ReservationReleaseResult.RELEASED, false));

        var observer = admission.admit(plan, endpoint);
        assertTrue(
                cancel
                        ? observer.cancel(false)
                        : observer.completeExceptionally(
                                new IllegalStateException("observer gone")));
        source.complete(new DecodeAdmission.PreemptionResult(reservation, false, "committed"));

        verify(plan).accept(endpoint, reservation);
        verify(endpoint, times(accepted ? 0 : 1))
                .release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
    }
}
