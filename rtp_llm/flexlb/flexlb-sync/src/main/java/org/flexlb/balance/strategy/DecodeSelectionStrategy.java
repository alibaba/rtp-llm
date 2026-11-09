package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;

import java.util.List;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * Ranks Decode candidates and coordinates registration of the selected request impact.
 */
@FunctionalInterface
public interface DecodeSelectionStrategy {
    /**
     * Rank candidates and register the winner. Refresh only conflicting candidates before retrying.
     * SUCCESS guarantees that the registration callback returned REGISTERED.
     */
    PlacementResult<DecodeResources.DecodeRoutingView, PlacementKey> selectAndRegister(
            Request request,
            SchedulingMetadata metadata,
            String group,
            List<DecodeResources.DecodeRoutingView> candidates,
            CapacityReader capacity,
            UnaryOperator<DecodeResources.DecodeRoutingView> refresh,
            Function<DecodeResources.DecodeRoutingView, PlacementResult.Registration> register);

    @FunctionalInterface
    interface CapacityReader {
        /** Null means the selected generation is no longer available. */
        DecodeResources.AdmissionSummary read(DecodeResources.DecodeRoutingView expected);
    }
}
