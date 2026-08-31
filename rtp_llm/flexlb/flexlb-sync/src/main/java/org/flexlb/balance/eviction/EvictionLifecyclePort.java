package org.flexlb.balance.eviction;

import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.concurrent.CompletableFuture;

/** Request lifecycle operations consumed only by eviction admission. */
public interface EvictionLifecyclePort {

    boolean isAdmissionOpen(
            long requestId, CompletableFuture<?> exactFuture);

    AdmissionMutation claimAdmissionMutation(
            long requestId, CompletableFuture<?> exactFuture);

    void finishYielded(ScheduledRequest victim, String detail);

    void finishYieldedReservation(
            long requestId, long reservationToken, String detail);
}
