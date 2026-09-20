package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint.AdmissionSnapshot;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;

import java.util.Collection;

/** Classify Decode admission rejection from the capacity snapshot used by the decision. */
public final class AdmissionFailureClassifier {

    private AdmissionFailureClassifier() {
    }

    public static Response classifyDecode(int incomingPriority, long hardKvTokens, long concurrencyLimit,
                                          Collection<AdmissionSnapshot> endpoints) {
        if (endpoints == null || endpoints.isEmpty()) {
            return Response.error(StrategyErrorType.NO_DECODE_WORKER,
                    AdmissionRejectReason.UNSPECIFIED, "no Decode endpoints available for admission");
        }
        Response common = null;
        boolean unanimous = true;
        for (AdmissionSnapshot endpoint : endpoints) {
            long slotDeficit = concurrencyLimit > 0 ? Math.max(0, endpoint.engineLoad() + 1L - concurrencyLimit) : 0;
            long kvDeficit = endpoint.kvTotal() > 0 ? Math.max(0, hardKvTokens - endpoint.kvAvailable()) : 0;
            Response failure;
            if ((endpoint.kvTotal() > 0 && hardKvTokens > endpoint.kvTotal())
                    || (slotDeficit <= 0 && kvDeficit <= 0)) {
                // The request cannot fit, or only the routing watermark rejected it.
                // Capacity may also have recovered before the failure snapshot.
                failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
            } else {
                failure = classifyEndpoint(incomingPriority, endpoint, slotDeficit, kvDeficit);
            }
            if (failure.getCode() == StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode()) {
                return failure;
            }
            if (common == null) {
                common = failure;
            } else if (common.getCode() != failure.getCode()
                    || common.getAdmissionRejectReason() != failure.getAdmissionRejectReason()) {
                unanimous = false;
            }
        }
        return unanimous ? common : Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    private static Response classifyEndpoint(int incomingPriority,
                                             AdmissionSnapshot endpoint,
                                             long slotDeficit,
                                             long kvDeficit) {
        long lowerSlot = 0;
        long lowerKv = 0;
        long higherSlot = 0;
        long higherKv = 0;
        long sameSlot = 0;
        long sameKv = 0;
        long unattributedSlot = endpoint.slots(0);
        long unattributedKv = endpoint.kvTokens(0);
        for (int priority = 1; priority <= 100; priority++) {
            long slots = endpoint.slots(priority);
            long kv = endpoint.kvTokens(priority);
            if (priority < incomingPriority) {
                lowerSlot += slots;
                lowerKv += kv;
            } else if (priority > incomingPriority) {
                higherSlot += slots;
                higherKv += kv;
            } else {
                sameSlot += slots;
                sameKv += kv;
            }
        }

        // Aggregate engine load / KV can exceed the individually tracked occupants.
        // Missing entries carry no priority provenance either.
        unattributedSlot += Math.max(0L, endpoint.engineLoad()
                - lowerSlot - higherSlot - sameSlot - unattributedSlot);
        long occupiedKv = Math.max(0L, endpoint.kvTotal() - endpoint.kvAvailable());
        unattributedKv += Math.max(0L, occupiedKv
                - lowerKv - higherKv - sameKv - unattributedKv);

        long residualSlot = Math.max(0, slotDeficit - lowerSlot);
        long residualKv = Math.max(0, kvDeficit - lowerKv);
        long knownProtectedSlot = higherSlot + sameSlot;
        long knownProtectedKv = higherKv + sameKv;
        boolean slotAttributionComplete = residualSlot <= 0
                || knownProtectedSlot >= residualSlot;
        boolean kvAttributionComplete = residualKv <= 0
                || knownProtectedKv >= residualKv;
        boolean unattributedBlocksResidual =
                (residualSlot > knownProtectedSlot && unattributedSlot > 0)
                        || (residualKv > knownProtectedKv && unattributedKv > 0);
        boolean protectedOccupancyCoversResidual =
                slotAttributionComplete && kvAttributionComplete;
        boolean higherBlocksResidual = protectedOccupancyCoversResidual
                && ((residualSlot > 0 && higherSlot > 0)
                || (residualKv > 0 && higherKv > 0));
        boolean sameBlocksResidual = protectedOccupancyCoversResidual
                && ((residualSlot > 0 && sameSlot > 0)
                || (residualKv > 0 && sameKv > 0));
        if (unattributedBlocksResidual) {
            return Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE);
        }
        if (higherBlocksResidual) {
            return Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                    AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
        }
        if (sameBlocksResidual) {
            return Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                    AdmissionRejectReason.SAME_PRIORITY_AHEAD);
        }
        // Lower-priority occupancy alone does not prove that ownership and
        // cancellation policy allow reclaiming it. Without a proven priority
        // blocker, retain the capacity rejection.
        return Response.error(StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }
}
