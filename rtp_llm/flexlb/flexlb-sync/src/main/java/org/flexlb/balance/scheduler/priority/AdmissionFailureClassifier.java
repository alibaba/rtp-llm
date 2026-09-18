package org.flexlb.balance.scheduler.priority;

import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.List;

/** Classify Decode admission rejection from the capacity snapshot used by the decision. */
public final class AdmissionFailureClassifier {

    private AdmissionFailureClassifier() {
    }

    public static Response classifyDecode(PriorityRequestEnvelope incoming,
                                          List<DecodeEndpointSnapshot> endpoints) {
        if (endpoints == null || endpoints.isEmpty()) {
            return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER,
                    AdmissionRejectReason.UNSPECIFIED, "no Decode endpoints available for admission");
        }
        List<Response> failures = new ArrayList<>();
        for (DecodeEndpointSnapshot endpoint : endpoints) {
            if (endpoint.realKvTotal() > 0
                    && incoming.hardKvTokens() > endpoint.realKvTotal()) {
                // This request can never fit on this endpoint, irrespective
                // of occupant priority.  Consequently an unattributed
                // occupant on this endpoint is not causally relevant.
                failures.add(Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                        AdmissionRejectReason.RESOURCE_EXHAUSTED));
                continue;
            }
            long slotDeficit = EvictionPlanner.slotDeficit(endpoint);
            long kvDeficit = EvictionPlanner.kvDeficit(incoming, endpoint);
            if (slotDeficit <= 0 && kvDeficit <= 0) {
                // The route failed while this endpoint snapshot has capacity;
                // the snapshot cannot prove a causal blocker.
                failures.add(Response.error(StrategyErrorType.BATCH_DISPATCH_FAILED, AdmissionRejectReason.UNSPECIFIED,
                        "Decode placement failed without a slot or KV capacity deficit"));
                continue;
            }
            failures.add(classifyEndpoint(incoming, endpoint, slotDeficit, kvDeficit));
        }
        // One causally blocked endpoint with missing priority provenance makes
        // cluster attribution unavailable. Folding it into 8431 would claim a
        // fully attributed resource failure that the aggregate snapshot cannot
        // prove, even when other endpoints have typed or resource causes.
        for (Response failure : failures) {
            if (failure.getCode() == StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode()) {
                return failure;
            }
        }
        for (Response failure : failures) {
            if (failure.getCode() == StrategyErrorType.BATCH_DISPATCH_FAILED.getErrorCode()) {
                return failure;
            }
        }
        Response common = failures.get(0);
        boolean unanimous = failures.stream().allMatch(failure ->
                failure.getCode() == common.getCode()
                        && failure.getAdmissionRejectReason() == common.getAdmissionRejectReason());
        return unanimous ? common : Response.error(StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    private static Response classifyEndpoint(PriorityRequestEnvelope incoming,
                                             DecodeEndpointSnapshot endpoint,
                                             long slotDeficit,
                                             long kvDeficit) {
        List<DecodeRequestSnapshot> occupants = new ArrayList<>(endpoint.reserved());
        occupants.addAll(endpoint.accepted());
        occupants.addAll(endpoint.running());

        long lowerSlot = 0;
        long lowerKv = 0;
        long higherSlot = 0;
        long higherKv = 0;
        long sameSlot = 0;
        long sameKv = 0;
        long unattributedSlot = 0;
        long unattributedKv = 0;
        for (DecodeRequestSnapshot occupant : occupants) {
            // slotDeficit is derived from engineLoad, so an N2 reservation
            // that is still queued on Prefill cannot explain that deficit.
            // It does reserve hard KV and therefore remains relevant to the
            // KV dimension.
            boolean contributesSlot = !occupant.queued();
            boolean contributesKv = occupant.kvTokens() > 0;
            if (!contributesSlot && !contributesKv) {
                continue;
            }
            if (!occupant.priorityKnown()
                    || !PriorityNormalizer.isValid(occupant.priority())) {
                if (contributesSlot) {
                    unattributedSlot++;
                }
                if (contributesKv) {
                    unattributedKv += occupant.kvTokens();
                }
                continue;
            }
            if (occupant.priority() < incoming.priority()) {
                if (contributesSlot) {
                    lowerSlot++;
                }
                if (contributesKv) {
                    lowerKv += occupant.kvTokens();
                }
            } else if (occupant.priority() > incoming.priority()) {
                if (contributesSlot) {
                    higherSlot++;
                }
                if (contributesKv) {
                    higherKv += occupant.kvTokens();
                }
            } else {
                // It is already admitted in this immutable snapshot, so its
                // admission sequence is necessarily ahead of the incoming.
                if (contributesSlot) {
                    sameSlot++;
                }
                if (contributesKv) {
                    sameKv += occupant.kvTokens();
                }
            }
        }

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
