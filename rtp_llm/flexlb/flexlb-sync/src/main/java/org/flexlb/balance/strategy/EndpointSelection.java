package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/** Result of selecting one role for a queue placement attempt. */
public record EndpointSelection(
        SelectedRole endpoint,
        RoleType blockerRole) {

    public EndpointSelection {
        if ((endpoint == null) == (blockerRole == null)) {
            throw new IllegalArgumentException(
                    "selection requires exactly one endpoint or blocker role");
        }
    }

    public static EndpointSelection selected(SelectedRole selected) {
        return new EndpointSelection(
                Objects.requireNonNull(selected, "selected"), null);
    }

    public static EndpointSelection unavailable(RoleType blockerRole) {
        return new EndpointSelection(
                null, Objects.requireNonNull(blockerRole, "blockerRole"));
    }

    public boolean selected() {
        return endpoint != null;
    }
}
