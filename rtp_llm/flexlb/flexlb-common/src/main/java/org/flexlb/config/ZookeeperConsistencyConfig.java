package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class ZookeeperConsistencyConfig implements ConsistencyConfig {

    private String connectString;
    private int sessionTimeoutMs = 30_000;
    private int connectionTimeoutMs = 30_000;
    private long masterRefreshIntervalMs = 5000L;
}
