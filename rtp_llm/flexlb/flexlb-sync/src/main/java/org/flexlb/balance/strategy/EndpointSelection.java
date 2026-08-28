package org.flexlb.balance.strategy;

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

    /** No endpoint ownership was acquired; retry after this role changes. */
    record Missed(RoleType blockerRole) implements EndpointSelection {
        public Missed {
            Objects.requireNonNull(blockerRole, "blockerRole");
        }
    }

    public static EndpointSelection selected(SelectedRole selected) {
        return new Selected(selected);
    }

    public static EndpointSelection unavailable(RoleType blockerRole) {
        return new Missed(Objects.requireNonNull(
                blockerRole, "blockerRole"));
    }
}
