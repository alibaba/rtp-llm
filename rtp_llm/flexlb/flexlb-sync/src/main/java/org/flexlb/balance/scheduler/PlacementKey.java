package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/** Logical capacity domain. A null group means any group for the role. */
public record PlacementKey(RoleType role, String group) {

    public PlacementKey {
        Objects.requireNonNull(role, "role");
        if (group != null && group.isBlank()) {
            group = null;
        }
    }

    public static PlacementKey anyGroup(RoleType role) {
        return new PlacementKey(role, null);
    }
}
