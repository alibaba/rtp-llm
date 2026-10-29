package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.RequestContext;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;

class DecodePreemptionHandoffTest {
    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void preemptionDoesNotOwnPlacementPlan(boolean cancel) {
        var admission = DecodePreemptionTestSupport.mock();
        var context = mock(RequestContext.class);
        var endpoint = mock(DecodeEndpoint.class);
        var source = new CompletableFuture<DecodePreemption.PreemptionResult>();
        doReturn(source).when(admission).tryReclaim(context, endpoint);

        var observer = admission.tryReclaim(context, endpoint);
        assertTrue(
                cancel
                        ? observer.cancel(false)
                        : observer.completeExceptionally(
                                new IllegalStateException("observer gone")));
        assertSame(source, observer);
    }
}
