package org.flexlb.dispatcher;

import org.flexlb.dispatcher.DispatchConfig.FeAllocation;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.core.env.MapPropertySource;
import org.springframework.core.env.StandardEnvironment;
import org.springframework.mock.env.MockEnvironment;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

@ExtendWith(SystemStubsExtension.class)
class DispatchConfigTest {
    @SystemStub
    private EnvironmentVariables credentials = new EnvironmentVariables("DISPATCH_ROUTING_TOKEN", "");

    @Test
    void dispatcherEnvironmentUsesExistingBindingAndValidation() {
        credentials.set("DISPATCH_FE_POOL_SERVICE_ID", "fe");
        credentials.set("DISPATCH_SUB_BATCH", "size:7");
        credentials.set("DISPATCH_FE_ALLOCATION", "local");
        credentials.set("DISPATCH_PRE_ASSIGN_BE", "false");
        credentials.set("DISPATCH_BATCH_TIMEOUT_MS", "1234");
        credentials.set("DISPATCH_BODY_READ_MARGIN_MS", "5678");
        credentials.set("DISPATCH_DISCOVERY_FAILURE_GRACE_MS", "0");
        credentials.set("DISPATCH_PROBE_PATH", "/health");
        credentials.set("DISPATCH_MAX_AGGREGATE_REQUEST_BYTES", "1000000");
        credentials.set("DISPATCH_MAX_AGGREGATE_RESPONSE_BYTES", "2000000");
        credentials.set("DISPATCH_ROUTING_TOKEN", "secret");
        credentials.set("DISPATCH_CONFIG", "{\"subBatch\":\"count:99\"}");
        credentials.set("SERVER_PORT", "12345");
        MockEnvironment env = environment();
        DispatchConfig cfg = DispatcherConfiguration.loadAndValidate(env);
        assertEquals("fe", cfg.getFePoolServiceId());
        assertEquals(new SubBatchSpec(SubBatchSpec.Mode.SIZE, 7), cfg.getSubBatchSpec());
        assertEquals(FeAllocation.LOCAL, cfg.getFeAllocation());
        assertFalse(cfg.isPreAssignBe());
        assertEquals(1234, cfg.getBatchTimeoutMs());
        assertEquals(5678, cfg.getBodyReadMarginMs());
        assertEquals(0, cfg.getDiscoveryFailureGraceMs());
        assertEquals("/health", cfg.getProbePath());
        assertEquals(1000000, cfg.getMaxAggregateRequestBytes());
        assertEquals(2000000, cfg.getMaxAggregateResponseBytes());
        assertEquals("secret", cfg.getTrustedRoutingToken());
        assertFalse(env.containsProperty("dispatch.routing-token"));
        assertFalse(env.containsProperty("dispatch.config"));
        assertFalse(env.containsProperty("server.port"));
        assertFalse(JsonUtils.toString(cfg).contains("secret"));
        credentials.set("DISPATCH_BATCH_TIMEOUT_MS", "not-a-number");
        assertThrows(RuntimeException.class, () -> DispatcherConfiguration.loadAndValidate(environment()));
        credentials.set("DISPATCH_BATCH_TIMEOUT_MS", "0");
        assertThrows(IllegalArgumentException.class, () -> DispatcherConfiguration.loadAndValidate(environment()));
    }

    private MockEnvironment environment() {
        MockEnvironment env = new MockEnvironment();
        env.getPropertySources().addFirst(new MapPropertySource(
                StandardEnvironment.SYSTEM_ENVIRONMENT_PROPERTY_SOURCE_NAME, Map.of()));
        new DispatchEnvironmentPostProcessor().postProcessEnvironment(env, null);
        return env;
    }

    @Test
    void nativePropertiesAndEnvironmentCredentialUseOneValidatedConfiguration() {
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> load(Map.of("dispatch.fe-pool-service-id", "fe")))
                .getMessage().contains("DISPATCH_ROUTING_TOKEN"));
        credentials.set("DISPATCH_ROUTING_TOKEN", "secret");
        DispatchConfig defaults = load(Map.of("dispatch.fe-pool-service-id", "fe"));
        assertEquals("count:5", defaults.getSubBatch());
        assertEquals(FeAllocation.MASTER, defaults.getFeAllocation());
        assertTrue(defaults.isPreAssignBe());
        assertEquals("/frontend_health", defaults.getProbePath());
        assertEquals("secret", defaults.getTrustedRoutingToken());
        assertFalse(JsonUtils.toString(defaults).contains("secret"));
        credentials.set("DISPATCH_ROUTING_TOKEN", "");
        DispatchConfig cfg = load(Map.of(
                "dispatch.fe-pool-service-id", "fe", "dispatch.sub-batch", "size:7",
                "dispatch.fe-allocation", "local", "dispatch.pre-assign-be", "false"));
        assertEquals("fe", cfg.getFePoolServiceId());
        assertEquals(new SubBatchSpec(SubBatchSpec.Mode.SIZE, 7), cfg.getSubBatchSpec());
        assertEquals(FeAllocation.LOCAL, cfg.getFeAllocation());
        assertFalse(cfg.isPreAssignBe());
        assertEquals("", cfg.getTrustedRoutingToken());
    }

    @ParameterizedTest
    @CsvSource({"batch-timeout-ms,0", "body-read-margin-ms,-1", "discovery-failure-grace-ms,-1", "max-aggregate-request-bytes,0",
            "max-aggregate-response-bytes,0", "fe-allocation,typo", "sub-batch,count:0", "pre-assign-be,true",
            "probe-path,health", "probe-path,//host/health", "probe-path,https://host/health", "probe-path,/health#fragment"})
    void invalidConfigurationFailsAtStartup(String key, String value) {
        MockEnvironment env = new MockEnvironment().withProperty("dispatch.fe-pool-service-id", "fe")
                .withProperty("dispatch.pre-assign-be", "false")
                .withProperty("dispatch." + key, value);
        assertThrows(RuntimeException.class, () -> DispatcherConfiguration.loadAndValidate(env));
    }

    private DispatchConfig load(Map<String, Object> properties) {
        MockEnvironment env = new MockEnvironment();
        env.getPropertySources().addFirst(new MapPropertySource("test", properties));
        return DispatcherConfiguration.loadAndValidate(env);
    }
}
