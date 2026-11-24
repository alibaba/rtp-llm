package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillCostOrderConfigTest {
    private static final String SINGLE_NON_BATCH = """
            {"scheduler":{"decision":{"type":"SINGLE"},"ordering":%s},
             "dispatcher":{"type":"NON_BATCH"%s}}
            """;

    @Test
    void defaultsToDisabled() {
        assertFalse(ConfigTestFixtures.parse("{}").getDispatcher().isPrefillCostOrder());
    }

    @ParameterizedTest
    @ValueSource(strings = {"FIFO", "PRIORITY"})
    void dispatchCostOrderIsIndependentOfGlobalOrdering(String ordering) {
        var config = ConfigTestFixtures.parse(SINGLE_NON_BATCH.formatted(
                "{\"type\":\"" + ordering + "\"}", ",\"prefillCostOrder\":true"));
        assertEquals(QueueOrderingConfig.Type.valueOf(ordering), config.queueScheduler().getOrdering().getType());
        assertTrue(config.getDispatcher().isPrefillCostOrder());
    }

    @ParameterizedTest
    @ValueSource(strings = {"{\"tokenBalance\":{}}", "{\"cacheAffinity\":{\"maxExtraTtftMs\":1000}}"})
    void dispatchCostOrderSupportsExistingPrefillRoutingPolicies(String prefill) {
        var config = ConfigTestFixtures.parse("""
                {"scheduler":{"decision":{"type":"SINGLE"}},"dispatcher":{"type":"NON_BATCH","prefillCostOrder":true},
                 "router":{"roles":{"prefill":%s}}}
                """.formatted(prefill));
        assertTrue(config.getDispatcher().isPrefillCostOrder());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void legacyValueMigratesToDispatcherAndSerializesOnlyThere(boolean enabled) {
        String legacy = "{\"prefillCostOrder\":" + enabled + "}";
        var config = ConfigTestFixtures.parse(SINGLE_NON_BATCH.formatted(legacy, ""));
        assertEquals(enabled, config.getDispatcher().isPrefillCostOrder());

        JsonNode effective = new ObjectMapper().valueToTree(config);
        assertTrue(effective.at("/dispatcher/prefillCostOrder").isBoolean());
        assertEquals(enabled, effective.at("/dispatcher/prefillCostOrder").booleanValue());
        assertTrue(effective.at("/scheduler/ordering/prefillCostOrder").isMissingNode());
        assertEquals(1, effective.findValues("prefillCostOrder").size());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void duplicateSettingsMustAgree(boolean legacy) {
        String ordering = "{\"prefillCostOrder\":" + legacy + "}";
        assertEquals(legacy, ConfigTestFixtures.parse(SINGLE_NON_BATCH.formatted(
                ordering, ",\"prefillCostOrder\":" + legacy)).getDispatcher().isPrefillCostOrder());
        var error = assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(
                SINGLE_NON_BATCH.formatted(ordering, ",\"prefillCostOrder\":" + !legacy)));
        assertTrue(error.getMessage().contains("dispatcher.prefillCostOrder"), error::getMessage);
    }

    @Test
    void enabledRequiresQueueSingleAndNonBatch() {
        for (String patch : new String[]{
                """
                {"scheduler":{"type":"DIRECT"},"dispatcher":{"type":"NON_BATCH","prefillCostOrder":true}}
                """,
                """
                {"scheduler":{"decision":{"type":"FIXED_WINDOW"}},"dispatcher":{"type":"NON_BATCH","prefillCostOrder":true}}
                """,
                """
                {"scheduler":{"decision":{"type":"SINGLE"}},"dispatcher":{"type":"BATCH","prefillCostOrder":true}}
                """,
                """
                {"scheduler":{"ordering":{"prefillCostOrder":true}},"dispatcher":{"type":"NON_BATCH"}}
                """}) {
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(patch), patch);
        }
    }

    @Test
    void legacyAliasDoesNotRelaxDocumentValidation() {
        assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse("""
                {"scheduler":{"type":"DIRECT","ordering":{"prefillCostOrder":false}},"dispatcher":{"type":"NON_BATCH"}}
                """));
        for (String invalid : new String[]{"null", "1", "\"true\""}) {
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(
                    SINGLE_NON_BATCH.formatted("{\"prefillCostOrder\":" + invalid + "}", "")), invalid);
            assertThrows(ConfigValidationException.class, () -> ConfigTestFixtures.parse(
                    SINGLE_NON_BATCH.formatted("{}", ",\"prefillCostOrder\":" + invalid)), invalid);
        }
    }
}
