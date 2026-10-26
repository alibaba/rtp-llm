package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestContext;

import java.util.concurrent.CompletableFuture;

/** Keep the real reservation handoff when tests control only the asynchronous capacity result. */
public final class DecodeAdmissionTestSupport {
    private DecodeAdmissionTestSupport() {}

    public static DecodeAdmission mock() {
        var admission = org.mockito.Mockito.mock(DecodeAdmission.class);
        org.mockito.Mockito.when(
                        admission.admit(
                                org.mockito.ArgumentMatchers.any(),
                                org.mockito.ArgumentMatchers.any()))
                .thenCallRealMethod();
        return admission;
    }

    public static CompletableFuture<DecodeAdmission.PreemptionResult> reclaim(
            DecodeAdmission admission, RequestContext context, WorkerEndpoint endpoint) {
        return admission.tryReclaim(context, endpoint);
    }
}
