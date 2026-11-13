package org.flexlb.config;

import java.net.Inet4Address;
import java.util.Objects;

/**
 * Per-JVM IPv4 loopback address for a locally launched Mock Master.
 * The Mock launcher registers it to share one address between gRPC binding and ZooKeeper
 * participant identity. Production deployments use the system host identity.
 */
public record MasterLoopbackAddress(Inet4Address address) {
    public MasterLoopbackAddress {
        Objects.requireNonNull(address, "Master loopback address");
        if (!address.isLoopbackAddress()) {
            throw new IllegalArgumentException(
                    "Local Mock Master requires an IPv4 loopback address");
        }
    }
}
