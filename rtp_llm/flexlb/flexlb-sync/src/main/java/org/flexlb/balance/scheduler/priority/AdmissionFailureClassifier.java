package org.flexlb.balance.scheduler.priority;

import org.flexlb.dao.loadbalance.StrategyErrorType;
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
            if (endpoint.realKvTotal() > 0
                    && incoming.hardKvTokens() > endpoint.realKvTotal()) {
                // This request can never fit on this endpoint, irrespective
                // of occupant priority.  Consequently an unattributed
                // occupant on this endpoint is not causally relevant.
                failures.add(new EndpointFailure(AdmissionFailure.resourceExhausted()));
                continue;
            }
            long slotDeficit = EvictionPlanner.slotDeficit(endpoint);
            long kvDeficit = EvictionPlanner.kvDeficit(incoming, endpoint);
            if (slotDeficit <= 0 && kvDeficit <= 0) {
                // The route failed while this endpoint snapshot has capacity;
                // the snapshot cannot prove a causal blocker.
                failures.add(new EndpointFailure(AdmissionFailure.resourceExhausted()));
                continue;
            }
            failures.add(classifyEndpoint(incoming, endpoint, slotDeficit, kvDeficit));
        }
        // One causally blocked endpoint with missing priority provenance makes
        // cluster attribution unavailable. Folding it into 8431 would claim a
        // fully attributed resource failure that the aggregate snapshot cannot
        // prove, even when other endpoints have typed or resource causes.
        for (EndpointFailure endpoint : failures) {
            if (endpoint.failure().errorType() ==
                    StrategyErrorType.ADMISSION_UNAVAILABLE) {
                return endpoint.failure();
            }
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
        int higher = 0;
        int same = 0;
        boolean hasUnattributedOccupant = false;
        for (QueuedRequestSnapshot occupant : queue.items()) {
            if (!QueuedRequestSnapshot.PREFILL_QUEUED.equals(occupant.state())) {
                continue;
            }
            if (!PriorityNormalizer.hasPriority(occupant.priority())) {
                hasUnattributedOccupant = true;
            } else if (occupant.priority() < incoming.priority()) {
                lower++;
            } else if (occupant.priority() > incoming.priority()) {
                higher++;
            } else {
                same++;
            }
        }
        int residual = Math.max(0, deficit - lower);
        if (deficit <= 0 || residual <= 0) {
            return AdmissionFailure.resourceExhausted();
        }
        // An unprioritized occupant is causal only when the known
        // higher/same-priority occupants cannot already cover the residual
        // queue deficit.  Merely being present must not erase a proven typed
        // blocker.
        int knownProtected = higher + same;
        if (hasUnattributedOccupant && knownProtected < residual) {
            return AdmissionFailure.priorityAttributionUnavailable();
        }
        if (higher > 0) {
            return AdmissionFailure.higherPriorityAhead();
        }
        if (same > 0) {
            return AdmissionFailure.samePriorityAhead();
        }
        return AdmissionFailure.resourceExhausted();
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
            List<QueuedRequestSnapshot> itemsAhead) {
        boolean higher = false;
        boolean same = false;
        boolean hasUnattributedOccupant = false;
        if (itemsAhead != null) {
            for (QueuedRequestSnapshot occupant : itemsAhead) {
                if (!QueuedRequestSnapshot.PREFILL_QUEUED.equals(occupant.state())) {
                    continue;
                }
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

    private static EndpointFailure classifyEndpoint(PriorityRequestEnvelope incoming,
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
                    || !PriorityNormalizer.hasPriority(occupant.priority())) {
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
        AdmissionFailure failure;
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
            failure = AdmissionFailure.priorityAttributionUnavailable();
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
