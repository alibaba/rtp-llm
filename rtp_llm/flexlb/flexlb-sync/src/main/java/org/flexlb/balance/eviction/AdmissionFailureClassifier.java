package org.flexlb.balance.eviction;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.util.PriorityNormalizer;

import java.util.List;

/** Pure causal classification for a timed-out priority queue entry. */
public final class AdmissionFailureClassifier {

    private AdmissionFailureClassifier() {
    }

    /**
     * Classify a priority-admitted request that timed out after it had already
     * entered the selected Prefill queue.
     *
     * <p>{@code itemsAhead} is the exact queue prefix from the timeout
     * decision snapshot. A higher-priority request is the primary blocker
     * whenever one exists. Otherwise an earlier request with the same priority
     * proves FIFO blocking. If the causal prefix only contains an occupant
     * without priority provenance, attribution is unavailable. If none of
     * those cases applies, the selected route failed to provide dispatch or
     * engine-admission capacity before expiration and is classified as
     * resource exhaustion. Lower-priority items and items behind the request
     * cannot explain its wait.
     */
    public static AdmissionFailure classifyQueuedTimeout(
            int incomingPriority,
            List<DeliveryItem> itemsAhead) {
        boolean higher = false;
        boolean same = false;
        boolean hasUnattributedOccupant = false;
        if (itemsAhead != null) {
            for (DeliveryItem occupant : itemsAhead) {
                if (!PriorityNormalizer.hasPriority(occupant.priority())) {
                    hasUnattributedOccupant = true;
                } else if (occupant.priority() > incomingPriority) {
                    higher = true;
                } else if (occupant.priority() == incomingPriority) {
                    same = true;
                }
            }
        }
        if (higher) {
            return AdmissionFailure.higherPriorityAhead();
        }
        if (same) {
            return AdmissionFailure.samePriorityAhead();
        }
        if (hasUnattributedOccupant) {
            return AdmissionFailure.priorityAttributionUnavailable();
        }
        return AdmissionFailure.resourceExhausted();
    }

}
