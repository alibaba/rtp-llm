package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/**
 * Placement blocker or capacity event. A null group means any group for the role;
 * an endpoint identifies the observed worker for diagnostics and preemption.
 * Waiting requests use the role/group domain and select again after capacity changes.
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

    public static PlacementKey anyGroup(RoleType role) {
        return new PlacementKey(role, null, null);
    }

    public static PlacementKey exact(RoleType role, String group, String endpoint) {
        return new PlacementKey(role, group, Objects.requireNonNull(endpoint, "endpoint"));
    }

    /**
     * Wait for a fresh decision across the allowed role/group, not a reservation on one worker.
     */
    PlacementKey waitingDomain() {
        return endpoint == null ? this : new PlacementKey(role, group, null);
    }
}
