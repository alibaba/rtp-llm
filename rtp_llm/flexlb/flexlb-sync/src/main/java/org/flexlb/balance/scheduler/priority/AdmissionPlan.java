package org.flexlb.balance.scheduler.priority;

import java.util.List;

/**
 * An admission plan built against a {@link ClusterSnapshot}: the ordered set
 * of actions required to admit one incoming request.
 *
 * <p>Plans are validated (version check) and applied by {@code PlanCommitter}.
 * Phase 1 has a single implementation ({@link NormalPlacementPlan}); eviction
 * plans arrive in later phases.
 */
public interface AdmissionPlan {

    /** The incoming request this plan admits. */
    PriorityRequestEnvelope envelope();

    /** Ordered actions of this plan. */
    List<PlanAction> actions();
}
