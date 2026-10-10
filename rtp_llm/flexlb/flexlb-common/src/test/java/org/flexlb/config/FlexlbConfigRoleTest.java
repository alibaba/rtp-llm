package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class FlexlbConfigRoleTest {
    @Test
    void request_timeout_is_explicit_and_dispatcher_defaults_are_shared() {
        FlexlbConfig config = new FlexlbConfig();
        assertNull(config.getRequestLifecycle().getRequest().getTimeoutMs());
        assertEquals(2.0, config.getRequestLifecycle().getDecision().getLifetime());
        assertEquals(2, config.getDispatcher().getMaxInflightPerPrefillWorker());
        assertEquals(2, DispatcherConfig.nonBatch().getMaxInflightPerPrefillWorker());
    }
}
