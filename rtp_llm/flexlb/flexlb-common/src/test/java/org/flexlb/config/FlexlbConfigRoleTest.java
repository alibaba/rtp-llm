package org.flexlb.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FlexlbConfigRoleTest {
    @Test
    void kvcmQueryTypeIsExplicitAndRejectsUnknownSemantics() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        KvcmCacheMatchingConfig config = mapper.readValue(
                "{\"type\":\"KVCM\",\"queryType\":\"QT_PREFIX_MATCH\"}", KvcmCacheMatchingConfig.class);
        assertEquals(KvcmCacheMatchingConfig.CacheQueryType.QT_PREFIX_MATCH, config.getQueryType());
        assertThrows(com.fasterxml.jackson.databind.exc.InvalidFormatException.class,
                () -> mapper.readValue("{\"type\":\"KVCM\",\"queryType\":\"guess\"}", KvcmCacheMatchingConfig.class));
    }

    @Test
    void request_timeout_is_explicit_and_dispatcher_defaults_are_shared() {
        FlexlbConfig config = new FlexlbConfig();
        assertNull(config.getRequestLifecycle().getRequest().getTimeoutMs());
        assertEquals(2.0, config.getRequestLifecycle().getDecision().getLifetime());
        assertEquals(2, config.getDispatcher().getMaxInflightPerPrefillWorker());
        assertEquals(2, DispatcherConfig.nonBatch().getMaxInflightPerPrefillWorker());
    }
}
