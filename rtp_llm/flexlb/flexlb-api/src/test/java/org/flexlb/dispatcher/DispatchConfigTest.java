package org.flexlb.dispatcher;

import org.flexlb.dispatcher.DispatchConfig.FeAllocation;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.core.env.SystemEnvironmentPropertySource;
import org.springframework.mock.env.MockEnvironment;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DispatchConfigTest {
    @Test
    void defaultsAndEnvironmentOverridesUseOneValidatedConfiguration() {
        DispatchConfig defaults = load(Map.of("DISPATCH_FE_POOL_SERVICE_ID", "fe"));
        assertEquals("count:5", defaults.getSubBatch());
        assertEquals(FeAllocation.MASTER, defaults.getFeAllocation());
        assertFalse(defaults.isPreAssignBe());
        assertEquals("/frontend_health", defaults.getProbePath());
        DispatchConfig cfg = load(Map.of(
                "DISPATCH_FE_POOL_SERVICE_ID", "env", "DISPATCH_SUB_BATCH", "size:7",
                "DISPATCH_FE_ALLOCATION", "local", "DISPATCH_PRE_ASSIGN_BE", "true",
                "DISPATCH_ROUTING_TOKEN", "secret"));
        assertEquals("env", cfg.getFePoolServiceId());
        assertEquals(new SubBatchSpec(SubBatchSpec.Mode.SIZE, 7), cfg.getSubBatchSpec());
        assertEquals(FeAllocation.LOCAL, cfg.getFeAllocation());
        assertTrue(cfg.isPreAssignBe());
        assertEquals("secret", cfg.getTrustedRoutingToken());
        assertFalse(JsonUtils.toString(cfg).contains("secret"));
    }

    @ParameterizedTest
    @CsvSource({"batch-timeout-ms,0", "body-read-margin-ms,-1", "max-aggregate-request-bytes,0",
            "max-aggregate-response-bytes,0", "fe-allocation,typo", "sub-batch,count:0", "pre-assign-be,true"})
    void invalidConfigurationFailsAtStartup(String key, String value) {
        MockEnvironment env = new MockEnvironment().withProperty("dispatch.fe-pool-service-id", "fe")
                .withProperty("dispatch." + key, value);
        assertThrows(RuntimeException.class, () -> DispatcherConfiguration.loadAndValidate(env));
    }

    private DispatchConfig load(Map<String, Object> properties) {
        MockEnvironment env = new MockEnvironment();
        env.getPropertySources().addFirst(new SystemEnvironmentPropertySource("test", properties));
        return DispatcherConfiguration.loadAndValidate(env);
    }
}
