package org.flexlb.mock;

import org.flexlb.config.FlexlbConfig;

/** Workload values shared by API test fixtures. */
public final class TestFlexlbConfigs {
    public static FlexlbConfig create() {
        FlexlbConfig config = new FlexlbConfig();
        config.getDispatcher().setType(org.flexlb.config.DispatcherConfig.Type.BATCH);
        config.setDispatchTimeoutMs(3_600_000L);
        config.getInternalRuntime().setRequestInactivityTimeoutMs(3_600_000L);
        config.getInternalRuntime().setEngineVisibilityMultiplier(2.0);
        return config;
    }

    private TestFlexlbConfigs() { }
}
