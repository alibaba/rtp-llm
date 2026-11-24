package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PrefillTokenBalanceConfigTest {
    private FlexlbConfig parse(String balance) {
        return ConfigTestFixtures.parse("{\"router\":{\"roles\":{\"prefill\":{\"tokenBalance\":" + balance + "}}}}");
    }

    @Test void optInDefaultsAndValidation() {
        assertNull(ConfigTestFixtures.parse("{}").getRouter().getRoles().getPrefill().getTokenBalance());
        assertEquals(5.0, parse("{}").getRouter().getRoles().getPrefill().getTokenBalance().getMinPrefixHitPercent());
        assertEquals(50.0, parse("{\"minPrefixHitPercent\":50}")
                .getRouter().getRoles().getPrefill().getTokenBalance().getMinPrefixHitPercent());
        for (String value : new String[]{"{\"minPrefixHitPercent\":101}", "{\"minPrefixHitPercent\":-1}"}) {
            assertThrows(RuntimeException.class, () -> parse(value));
        }
    }

    @Test void engineQueueAdmissionDefaultsToTen() {
        var defaults = parse("{}").getRouter().getRoles().getPrefill().getTokenBalance();
        assertEquals(10L, defaults.getEngineQueueAdmissionThreshold());
        assertEquals(5_000L, defaults.getEngineQueueOverloadDurationMs());
        var enabled = parse("{\"engineQueueAdmissionThreshold\":0}")
                .getRouter().getRoles().getPrefill().getTokenBalance();
        assertEquals(0L, enabled.getEngineQueueAdmissionThreshold());
        assertThrows(RuntimeException.class, () -> parse("{\"engineQueueAdmissionThreshold\":-1}"));
        assertThrows(RuntimeException.class, () -> parse("{\"engineQueueOverloadDurationMs\":-1}"));
    }

    @Test void removedRescueAndHistoryOptionsAreRejected() {
        for (String value : new String[]{"{\"hotspotRescue\":{}}", "{\"enableWindow\":false}",
                "{\"enableWindow\":true}", "{\"windowMs\":10000}"}) {
            assertThrows(ConfigValidationException.class, () -> parse(value));
        }
    }

    @Test void hotspotReleaseDefaultsToEnabledAndValidatesTuning() {
        var defaults = ConfigTestFixtures.parse("{}")
                .getRouter().getRoles().getPrefill().getHotspotRelease();
        assertEquals(true, defaults.isEnabled());
        assertEquals(1.5, defaults.getOutlierFactor());
        assertEquals(0L, defaults.getSustainedPressureMs());
        assertEquals(10, defaults.getMaxExtraComputePermille());
        assertEquals(5, hotspotParse("{\"maxExtraComputePermille\":5}").getMaxExtraComputePermille());
        assertEquals(0, hotspotParse("{\"maxExtraComputePermille\":0}").getMaxExtraComputePermille());
        assertThrows(ConfigValidationException.class, () -> hotspotParse("{\"maxExtraComputePermille\":-1}"));
        assertThrows(ConfigValidationException.class, () -> hotspotParse("{\"maxExtraComputePermille\":1001}"));
        assertEquals(10L, defaults.getSetpointFloor());
        assertEquals(1000L, defaults.getComputeTokenFloor());
        assertEquals(4096L, hotspotParse("{\"computeTokenFloor\":4096}").getComputeTokenFloor());
        assertThrows(ConfigValidationException.class, () -> hotspotParse("{\"computeTokenFloor\":-1}"));
        assertEquals(false, hotspotParse("{\"enabled\":false}").isEnabled());
        assertThrows(ConfigValidationException.class,
                () -> hotspotParse("{\"outlierFactor\":0.5}"));
        assertThrows(ConfigValidationException.class,
                () -> hotspotParse("{\"sustainedPressureMs\":-1}"));
        assertThrows(ConfigValidationException.class,
                () -> hotspotParse("{\"setpointFloor\":-1}"));
    }

    /** The pre-1.5-fleet-release boolean switch was replaced by the object form; keep it loud. */
    @Test void removedHotspotReleaseBooleanSwitchIsRejected() {
        assertThrows(ConfigValidationException.class,
                () -> ConfigTestFixtures.parse(
                        "{\"router\":{\"roles\":{\"prefill\":{\"hotspotReleaseEnabled\":false}}}}"));
    }

    private RoutingConfig.HotspotReleaseConfig hotspotParse(String release) {
        return ConfigTestFixtures.parse(
                        "{\"router\":{\"roles\":{\"prefill\":{\"hotspotRelease\":" + release + "}}}}")
                .getRouter().getRoles().getPrefill().getHotspotRelease();
    }
}
