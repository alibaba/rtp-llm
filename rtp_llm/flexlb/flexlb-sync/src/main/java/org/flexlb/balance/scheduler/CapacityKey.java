package org.flexlb.balance.scheduler;

import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/**
 * Identifies the role, group and optional worker for a capacity wait or change notification.
 * A null group means any group for the role;
 * an endpoint identifies the observed worker for diagnostics and preemption.
 * Waiting requests use the role/group domain and select again after capacity changes.
 */
public record CapacityKey(RoleType role, String group, String endpoint) {
    public CapacityKey {
        Objects.requireNonNull(role, "role");
        if (group != null && group.isBlank()) {
            group = null;
        }
        if (endpoint != null && endpoint.isBlank()) {
            endpoint = null;
        }
    }

    public static CapacityKey anyGroup(RoleType role) {
        return new CapacityKey(role, null, null);
    }

    public static CapacityKey exact(RoleType role, String group, String endpoint) {
        return new CapacityKey(role, group, Objects.requireNonNull(endpoint, "endpoint"));
    }

    /**
     * Wait for a fresh decision across the allowed role/group, not a reservation on one worker.
     */
    CapacityKey waitingDomain() {
        return endpoint == null ? this : new CapacityKey(role, group, null);
    }
}
