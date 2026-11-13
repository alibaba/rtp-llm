package org.flexlb.config.master;

import lombok.Getter;
import lombok.Setter;

/**
 * Configures the ZooKeeper connection and election parameters for master participation.
 */
@Getter
@Setter
public final class ZookeeperConsistencyConfig implements ConsistencyConfig {

    private String connectString;
    private int sessionTimeoutMs = 30_000;
    private int connectionTimeoutMs = 30_000;
    private long masterRefreshIntervalMs = 5000L;
}
