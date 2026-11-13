package org.flexlb.util.discovery;

import lombok.Getter;
import lombok.Setter;

/**
 * Runtime policy shared by service-discovery clients.
 */
@Getter
@Setter
public final class DiscoveryClientSettings {

    private int connectTimeoutMs = 500;
    private int readTimeoutMs = 500;
    private long pollIntervalMs = 1000L;
    private int maxIdleConnections = 5;
    private long keepAliveDurationMs = 300_000L;
}
