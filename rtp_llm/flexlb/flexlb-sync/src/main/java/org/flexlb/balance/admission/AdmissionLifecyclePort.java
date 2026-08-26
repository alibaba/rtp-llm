package org.flexlb.balance.admission;

import java.util.concurrent.CompletableFuture;

/** Canonical request-lifecycle boundary used by eviction admission. */
public interface AdmissionLifecyclePort {

    boolean isAdmissionOpen(
            long requestId, CompletableFuture<?> exactFuture);

    AdmissionMutation claimAdmissionMutation(
            long requestId, CompletableFuture<?> exactFuture);

    boolean bindAdmissionResources(
            long requestId,
            CompletableFuture<?> exactFuture,
            Runnable releasePermit,
            long acceptanceTimeoutMs);
}
