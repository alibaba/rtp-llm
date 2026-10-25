package org.flexlb.balance.endpoint.prefill.single;

import org.flexlb.balance.endpoint.prefill.LeaseState;
import org.flexlb.balance.endpoint.prefill.PrefillState;
import org.flexlb.balance.endpoint.prefill.RequestEntry;
import org.flexlb.balance.endpoint.prefill.Reservation;
import org.flexlb.balance.scheduler.ScheduledRequest;

public final class SingleReservation extends Reservation {
    public final String requestId;
    /* guarded by PrefillState.lock until the reservation commits */
    private long predictedWorkMs;

    public SingleReservation(PrefillState owner, RequestEntry originalOwner, String requestId, long predictedWorkMs) {
        super(owner, originalOwner);
        this.requestId = requestId;
        this.predictedWorkMs = boundedPrediction(predictedWorkMs);
    }

    public long predictedWorkMs() {
        return predictedWorkMs;
    }

    /**
     * Bind the fresh delivery prediction to this exact OPEN reservation.
     */
    public void updatePrediction(ScheduledRequest exactItem, long predictedMs) {
        owner.lock.lock();
        try {
            RequestEntry entry = owner.requests.get(requestId);
            if (state != LeaseState.OPEN
                    || entry == null
                    || !entry.activeIdentity(exactItem)
                    || entry.reservation != this) {
                throw new IllegalStateException(
                        "route prediction no longer owns ACTIVE request_id="
                                + requestId);
            }
            predictedWorkMs = boundedPrediction(predictedMs);
            owner.recordMutationUnderLock();
        } finally {
            owner.lock.unlock();
        }
    }

    private static long boundedPrediction(long predictedMs) {
        return Math.min(Integer.MAX_VALUE, Math.max(0L, predictedMs));
    }
}
