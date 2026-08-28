package org.flexlb.balance.strategy;

import java.util.Objects;

/** Result of selecting one role for a queue placement attempt. */
public record EndpointSelection(
        SelectedRole selected,
        boolean poolUnavailable) {

    public EndpointSelection {
        if (selected != null && poolUnavailable) {
            throw new IllegalArgumentException(
                    "a selected endpoint cannot be pool-unavailable");
        }
    }

    public static EndpointSelection selected(SelectedRole selected) {
        return new EndpointSelection(
                Objects.requireNonNull(selected, "selected"), false);
    }

    /** No request can use this pool on the observed capacity view. */
    public static EndpointSelection unavailablePool() {
        return new EndpointSelection(null, true);
    }

    /** This request cannot use the pool, but another request might. */
    public static EndpointSelection requestUnavailable() {
        return new EndpointSelection(null, false);
    }

    public boolean found() {
        return selected != null;
    }
}
