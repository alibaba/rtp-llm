package org.flexlb.balance.endpoint;

import org.flexlb.dao.route.RoleType;

/** Read-only node counts for monitoring. */
public interface EndpointStats {
    record Counts(int discovered, int published) {}

    int discoveredCount();

    Counts counts(RoleType role);
}
