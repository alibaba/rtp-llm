package org.flexlb.balance.scheduler.priority;

import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.List;

/** Pure causal classification over the same decode snapshot used to plan. */
public final class AdmissionFailureClassifier {

    private AdmissionFailureClassifier() {
    }

    public static AdmissionFailure classifyDecode(PriorityRequestEnvelope incoming,
                                                  List<DecodeEndpointSnapshot> endpoints) {
        if (endpoints == null || endpoints.isEmpty()) {
            return AdmissionFailure.resourceExhausted();
        }
        // A known hard-capacity miss on every endpoint is the one cluster-wide
        // resource fact that does not depend on victim ownership or policy.
        boolean allKnownTooSmall = endpoints.stream().allMatch(endpoint ->
                endpoint.realKvTotal() > 0
                        && incoming.hardKvTokens() > endpoint.realKvTotal());
        if (allKnownTooSmall) {
            return AdmissionFailure.resourceExhausted();
        }

        List<EndpointFailure> failures = new ArrayList<>();
        for (DecodeEndpointSnapshot endpoint : endpoints) {
            long slotDeficit = EvictionPlanner.slotDeficit(endpoint);
            long kvDeficit = EvictionPlanner.kvDeficit(incoming, endpoint);
            if (slotDeficit <= 0 && kvDeficit <= 0) {
                // The route failed while this endpoint snapshot has capacity;
                // the snapshot cannot prove a causal blocker.
                return AdmissionFailure.resourceExhausted();
            }
            failures.add(classifyEndpoint(incoming, endpoint, slotDeficit, kvDeficit));
        }
        AdmissionFailure common = failures.get(0).failure();
        boolean unanimous = failures.stream().allMatch(endpoint ->
                endpoint.failure().errorType() == common.errorType()
                        && endpoint.failure().reason() == common.reason());
        return unanimous ? common : AdmissionFailure.resourceExhausted();
    }

    public static AdmissionFailure classifyPrefill(PriorityRequestEnvelope incoming,
                                                   PrefillQueueSnapshot queue) {
        int deficit = queue.queueCapacity() > 0
                ? Math.max(0, queue.items().size() + 1 - queue.queueCapacity()) : 0;
        int lower = 0;
        boolean higher = false;
        boolean same = false;
        boolean unknown = false;
        for (QueuedRequestSnapshot occupant : queue.items()) {
            if (!QueuedRequestSnapshot.PREFILL_QUEUED.equals(occupant.state())) {
                continue;
            }
            if (!PriorityNormalizer.hasPriority(occupant.priority())) {
                unknown = true;
            } else if (occupant.priority() < incoming.priority()) {
                lower++;
            } else if (occupant.priority() > incoming.priority()) {
                higher = true;
            } else {
                same = true;
            }
        }
        int residual = Math.max(0, deficit - lower);
        if (deficit <= 0 || residual <= 0) {
            return AdmissionFailure.resourceExhausted();
        }
        if (unknown) {
            return AdmissionFailure.resourceExhausted();
        }
        if (higher) {
            return AdmissionFailure.higherPriorityAhead();
        }
        if (same) {
            return AdmissionFailure.samePriorityAhead();
        }
        return AdmissionFailure.resourceExhausted();
    }

    /**
     * Classify a priority request that reached its admission deadline after it
     * had already entered the selected Prefill queue.
     *
     * <p>{@code itemsAhead} is the exact queue prefix from the deadline
     * decision snapshot. A higher-priority request is the primary blocker
     * whenever one exists. Otherwise an earlier request with the same priority
     * proves FIFO blocking. If neither exists, the selected route failed to
     * provide dispatch/engine admission capacity within the budget and is
     * classified as resource exhaustion. Lower-priority items and items behind
     * the request cannot explain its wait.
     */
    public static AdmissionFailure classifyQueuedDeadline(
            int incomingPriority,
            List<QueuedRequestSnapshot> itemsAhead) {
        boolean higher = false;
        boolean same = false;
        if (itemsAhead != null) {
            for (QueuedRequestSnapshot occupant : itemsAhead) {
                if (!QueuedRequestSnapshot.PREFILL_QUEUED.equals(occupant.state())) {
                    continue;
                }
                if (occupant.priority() > incomingPriority) {
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
        return AdmissionFailure.resourceExhausted();
    }

    private static EndpointFailure classifyEndpoint(PriorityRequestEnvelope incoming,
                                                     DecodeEndpointSnapshot endpoint,
                                                     long slotDeficit,
                                                     long kvDeficit) {
        List<DecodeRequestSnapshot> occupants = new ArrayList<>(endpoint.reserved());
        occupants.addAll(endpoint.accepted());
        occupants.addAll(endpoint.running());

        long lowerSlot = 0;
        long lowerKv = 0;
        boolean higherSlot = false;
        boolean higherKv = false;
        boolean sameSlot = false;
        boolean sameKv = false;
        boolean unknownSlot = false;
        boolean unknownKv = false;
        for (DecodeRequestSnapshot occupant : occupants) {
            // Every snapshot occupant holds a Master admission slot. A
            // MASTER_QUEUED_NOT_DISPATCHED reservation has not reached the
            // engine yet, but it is still ahead of the incoming request in
            // the same authoritative ledger and therefore explains a slot
            // deficit (including same-priority FIFO blocking).
            boolean contributesSlot = true;
            boolean contributesKv = occupant.kvTokens() > 0;
            if (!contributesSlot && !contributesKv) {
                continue;
            }
            if (!occupant.priorityKnown()
                    || !PriorityNormalizer.hasPriority(occupant.priority())) {
                unknownSlot |= contributesSlot;
                unknownKv |= contributesKv;
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
                higherSlot |= contributesSlot;
                higherKv |= contributesKv;
            } else {
                // It is already admitted in this immutable snapshot, so its
                // admission sequence is necessarily ahead of the incoming.
                sameSlot |= contributesSlot;
                sameKv |= contributesKv;
            }
        }

        long residualSlot = Math.max(0, slotDeficit - lowerSlot);
        long residualKv = Math.max(0, kvDeficit - lowerKv);
        AdmissionFailure failure;
        boolean higherBlocksResidual = (residualSlot > 0 && higherSlot)
                || (residualKv > 0 && higherKv);
        boolean sameBlocksResidual = (residualSlot > 0 && sameSlot)
                || (residualKv > 0 && sameKv);
        boolean unknownBlocksResidual = (residualSlot > 0 && unknownSlot)
                || (residualKv > 0 && unknownKv);
        if (unknownBlocksResidual) {
            failure = AdmissionFailure.resourceExhausted();
        } else if (higherBlocksResidual) {
            failure = AdmissionFailure.higherPriorityAhead();
        } else if (sameBlocksResidual) {
            failure = AdmissionFailure.samePriorityAhead();
        } else if ((slotDeficit > 0 && lowerSlot >= slotDeficit)
                || (kvDeficit > 0 && lowerKv >= kvDeficit)) {
            // Lower-priority occupancy appears sufficient numerically, but
            // this snapshot cannot prove ownership homogeneity, cancel support
            // or policy gates. With no proven higher/same cause, fold the
            // control limitation into the broadened resource result rather
            // than claiming a physical allocation failure.
            failure = AdmissionFailure.resourceExhausted();
        } else {
            failure = AdmissionFailure.resourceExhausted();
        }
        return new EndpointFailure(failure);
    }

    private record EndpointFailure(AdmissionFailure failure) {
    }
}
