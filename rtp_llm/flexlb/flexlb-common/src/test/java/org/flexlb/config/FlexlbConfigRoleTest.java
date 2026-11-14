package org.flexlb.config;

import org.flexlb.config.ConfigTestSupport;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class FlexlbConfigRoleTest {
    @Test
    void dispatch_timeout_and_dispatcher_defaults_are_shared() {
        FlexlbConfig config = new FlexlbConfig();
        assertEquals(60000L, config.getDispatchTimeoutMs());
        assertEquals(2.0, config.getInternalRuntime().getEngineVisibilityMultiplier());
        assertEquals(128, config.getDispatcher().getMaxInflightPerPrefillWorker());
        assertEquals(128, ConfigTestSupport.nonBatch().getMaxInflightPerPrefillWorker());
    }
}
