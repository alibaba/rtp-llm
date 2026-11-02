package org.flexlb.mock;

import org.flexlb.config.FlexlbConfig;

/** Workload values shared by API test fixtures. */
public final class TestFlexlbConfigs {
    public static FlexlbConfig create() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(3_600_000L);
        config.getRequestLifecycle().getDecision().setLifetime(2.0);
        return config;
    }

    private TestFlexlbConfigs() { }
}
