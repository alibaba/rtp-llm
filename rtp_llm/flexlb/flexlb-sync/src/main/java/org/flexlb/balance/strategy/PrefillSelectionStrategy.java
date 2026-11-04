package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.function.LongSupplier;

/** Synchronous recommendation only; the caller validates identity and acquires resources. */
public interface PrefillSelectionStrategy {
    /** Synchronous snapshot access; only the selected input identity may be retained in the result. */
    interface CandidateInput {
        String address();

        String group();

        long generationId();

        long cacheBlockSize();

        RequestTimeEstimation.Inputs projectionInputs();

        /** Read-only current capacity. Selection never reserves a slot. */
        boolean requestCapacityAvailable();

        /** Exact priority counts, read only on a capacity failure. */
        AdmissionSummary admissionSummary(int priority);
    }

    record AdmissionSummary(
            long occupiedRequests,
            long requestsToRelease,
            long lowerPriorityRequests,
            long samePriorityRequests,
            long higherPriorityRequests,
            long unknownPriorityRequests) {}

    /**
     * Frozen scores; input must be the original candidate object, never a reconstructed identity.
     * ownershipVersion belongs to the snapshot used for scoring; negative TTFT means unknown.
     * No borrowed estimator result or asynchronous work may escape selection through this result.
     */
    record Selection<I extends CandidateInput>(
            String address,
            long generationId,
            long ownershipVersion,
            long projectedTtftMs,
            long prefillWorkMs,
            long cacheHitTokens,
            long routingCacheMatchTokens,
            long maximumRoutingCacheMatchTokens,
            String affinityReason,
            int workerCount,
            I input) {}

    <I extends CandidateInput> PlacementResult<Selection<I>, PlacementKey> select(
            Request request,
            SchedulingMetadata metadata,
            RoleType role,
            String group,
            List<I> inputs,
            LongSupplier clock);
}
