package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.PlacementBlockScope;
import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/** Closed result of selecting one role for a queue placement attempt. */
public sealed interface EndpointSelection
        permits EndpointSelection.Selected, EndpointSelection.Missed {

    /** One exact endpoint generation is owned by the caller. */
    record Selected(SelectedRole endpoint) implements EndpointSelection {
        public Selected {
            Objects.requireNonNull(endpoint, "endpoint");
        }
    }

    /** No endpoint ownership was acquired; scope states what the miss proves. */
    record Missed(
            PlacementBlockScope scope,
            RoleType blockerRole) implements EndpointSelection {
        public Missed {
            Objects.requireNonNull(scope, "scope");
            Objects.requireNonNull(blockerRole, "blockerRole");
        }
    }

    public static EndpointSelection selected(SelectedRole selected) {
        return new Selected(selected);
    }

    /** No request can use this pool on the observed capacity view. */
    public static EndpointSelection unavailablePool(RoleType blockerRole) {
        return new Missed(
                PlacementBlockScope.POOL_WIDE,
                Objects.requireNonNull(blockerRole, "blockerRole"));
    }

    /** The miss has no pool-wide proof; other candidates must remain eligible. */
    public static EndpointSelection requestUnavailable(RoleType blockerRole) {
        return new Missed(
                PlacementBlockScope.LIMITED,
                Objects.requireNonNull(blockerRole, "blockerRole"));
    }
}
