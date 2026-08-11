package org.flexlb.balance.endpoint;

import org.flexlb.dao.route.RoleType;

import java.util.Objects;

/**
 * Master-local identity of one published worker endpoint generation.
 *
 * <p>The generation is deliberately local to the master. It fences callbacks
 * and accounting belonging to an endpoint object that has already retired;
 * it is not a substitute for an engine-provided process incarnation id.
 */
public record EndpointId(RoleType role, String ipPort, long generation)
        implements Comparable<EndpointId> {

    public EndpointId {
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(ipPort, "ipPort");
        if (ipPort.isBlank()) {
            throw new IllegalArgumentException("ipPort must not be blank");
        }
        if (generation < 0) {
            throw new IllegalArgumentException("generation must be non-negative");
        }
    }

    @Override
    public int compareTo(EndpointId other) {
        int roleOrder = Integer.compare(role.ordinal(), other.role.ordinal());
        if (roleOrder != 0) {
            return roleOrder;
        }
        int addressOrder = ipPort.compareTo(other.ipPort);
        if (addressOrder != 0) {
            return addressOrder;
        }
        return Long.compare(generation, other.generation);
    }
}
