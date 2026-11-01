package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;

import java.util.List;

/** Returns the complete selection or failure; reading capacity does not reserve it. */
@FunctionalInterface
public interface DecodeSelectionStrategy {
    /** Candidates are borrowed and read-only; implementations must not retain them. */
    PlacementResult<DecodeResources.DecodeRoutingView, PlacementKey> select(
            Request request,
            SchedulingMetadata metadata,
            String group,
            List<DecodeResources.DecodeRoutingView> candidates,
            CapacityReader capacity);

    @FunctionalInterface
    interface CapacityReader {
        /** Null means the selected generation is no longer available. */
        DecodeResources.AdmissionSummary read(DecodeResources.DecodeRoutingView expected);
    }
}
