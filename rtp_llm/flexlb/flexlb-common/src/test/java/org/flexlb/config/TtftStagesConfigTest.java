package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TtftStagesConfigTest {
    @Test
    void parsesNewSettingsThroughExistingStrictConfigEntryPoint() {
        var config = ConfigService.parse("""
                {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}},
                 "observability":{"ttftStages":{"enabled":false,"maxRecords":2048}}}
                """);
        assertEquals(2048, config.getObservability().getTtftStages().getMaxRecords());
        assertFalse(config.getObservability().getTtftStages().isEnabled());
    }

    private static FlexlbConfig validConfig() {
        var config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        return config;
    }

    @Test
    void defaultsToCollectionWithoutAssumingHybridKvOrder() {
        var config = ConfigService.parse("""
                {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}}}
                """);
        assertTrue(config.getObservability().getTtftStages().isEnabled());
        assertFalse(config.getObservability().getTtftStages().isRemoteKvBeforeScheduler());
        assertDoesNotThrow(() -> FlexlbConfigValidator.validate(config));
    }

    @Test
    void enabledCollectionUsesModelServiceDiscoveryWithoutExplicitAddresses() {
        var config = validConfig();
        config.getObservability().getTtftStages().setEnabled(true);
        assertDoesNotThrow(() -> FlexlbConfigValidator.validate(config));
    }

    @Test
    void rejectsUnboundedPageSize() {
        var config = validConfig();
        config.getObservability().getTtftStages().setMaxRecords(4097);
        assertThrows(ConfigValidationException.class, () -> FlexlbConfigValidator.validate(config));
    }

    @Test
    void rejectsRemovedExplicitEndpointFieldInsteadOfSilentlyIgnoringIt() {
        for (String endpoints : new String[]{"[]", "[{\"host\":\"daemon\",\"port\":8080,\"role\":\"frontend\"}]"}) {
            String document = """
                    {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}},
                     "observability":{"ttftStages":{"enabled":true,"endpoints":%s}}}
                    """.formatted(endpoints);
            ConfigValidationException error =
                    assertThrows(ConfigValidationException.class, () -> ConfigService.parse(document));
            assertTrue(error.getMessage().contains("endpoints"));
        }
    }
}
