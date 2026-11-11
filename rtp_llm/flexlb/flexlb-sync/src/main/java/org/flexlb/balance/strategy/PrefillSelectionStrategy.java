package org.flexlb.balance.strategy;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.scheduler.CapacityKey;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.function.Function;
import java.util.function.LongSupplier;

/**
 * Ranks Prefill candidates and coordinates registration of the selected request impact.
 */
public interface PrefillSelectionStrategy {
    /**
     * Rank candidates and register the winner. Refresh only conflicting candidates before retrying.
     * SUCCESS guarantees that the registration callback returned REGISTERED.
     */
    <I extends CandidateInput> RoutingResult<Selection<I>, CapacityKey> selectAndRegister(
            Request request,
            SchedulingMetadata metadata,
            RoleType role,
            String group,
            List<I> inputs,
            LongSupplier clock,
            Function<Selection<I>, RoutingResult.Registration> register);

    /**
     * Synchronous snapshot access; only the selected input identity may be retained in the result.
     */
    interface CandidateInput {
        String address();

        String group();

        long generationId();

        long cacheBlockSize();

        String physicalAddress();

        int engineCount();

        RequestTimeEstimation.Inputs projectionInputs();

        /**
         * Read-only capacity probe; registration is a separate atomic endpoint operation.
         */
        boolean requestCapacityAvailable();

        /**
         * Explains a capacity failure without reserving resources.
         */
        AdmissionRejectReason explainAdmissionRejection(int priority);
    }

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
            I input,
            org.flexlb.cache.domain.CacheMatchResult cacheMatches,
            RequestTimeEstimation.Inputs projectionInputs) {}
}
