package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

public final class BatchReservation extends Reservation {
    final String headRequestId;
    final long batchId;
    /* guarded by PrefillState.lock; non-null only while OPEN */
    EndpointGenerationLifecycle.HandoffPermit generationHandoff;

    BatchReservation(PrefillState owner,
                     RequestEntry originalOwner,
                     String headRequestId,
                     long batchId,
                     EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        super(owner, originalOwner);
        this.headRequestId = headRequestId;
        this.batchId = batchId;
        this.generationHandoff = generationHandoff;
    }

    public long batchId() {
        return batchId;
    }

    /**
     * Atomically commit this exact batch lease.
     */
    public PrefillState.CommittedHandoff commit(List<ScheduledRequest> items, long predictedMs) {
        return owner.batchPrefill.commitBatch(this, items, predictedMs);
    }
}
