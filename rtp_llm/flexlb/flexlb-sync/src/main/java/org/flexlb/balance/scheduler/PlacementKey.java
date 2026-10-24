package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/**
 * Logical capacity domain. A null group means any group for the role; a
 * non-null endpoint identifies one exact worker within the group.
 */
public record PlacementKey(RoleType role, String group, String endpoint) {

    public PlacementKey {
        Objects.requireNonNull(role, "role");
        if (group != null && group.isBlank()) {
            group = null;
        }
        if (endpoint != null && endpoint.isBlank()) {
            endpoint = null;
        }
    }

    /** Exact waiters follow role/address across changes to the endpoint's group. */
    PlacementKey capacityDomain() {
        return endpoint() == null || group() == null ? this : exact(role(), null, endpoint());
    }

    public static PlacementKey anyGroup(RoleType role) {
        return new PlacementKey(role, null, null);
    }

    public static PlacementKey exact(
            RoleType role,
            String group,
            String endpoint) {
        return new PlacementKey(
                role, group, Objects.requireNonNull(endpoint, "endpoint"));
    }
}
