package org.flexlb.config;

import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillSchedulerConfigTest {
    @Test
    void defaultsToFlexlbAndRoundTripsTheActiveVariant() throws Exception {
        var defaults = ConfigTestFixtures.parse("{}");
        assertFalse(defaults.isNaviSchedPrefill());
        assertInstanceOf(FlexlbPrefillSchedulerConfig.class, defaults.getPrefillScheduler());
        assertFalse(
                ConfigTestFixtures.parse(
                                "{\"prefillScheduler\":"
                                        + JsonUtils.toStrictString(defaults.getPrefillScheduler())
                                        + "}")
                        .isNaviSchedPrefill());
        var delegated =
                ConfigTestFixtures.parse(
                        """
                {"prefillScheduler":{"type":"NAVI_SCHED",
                  "masterBizName":"master_biz"}}
                """);
        assertTrue(delegated.isNaviSchedPrefill());
        var restored =
                ConfigTestFixtures.parse(
                        "{\"prefillScheduler\":"
                                + JsonUtils.toStrictString(delegated.getPrefillScheduler())
                                + "}");
        var scheduler =
                assertInstanceOf(
                        NaviSchedPrefillSchedulerConfig.class, restored.getPrefillScheduler());
        assertEquals("master_biz", scheduler.getMasterBizName());
    }

    @Test
    void rejectsInactiveFieldsAndUnknownSchedulerTypes() {
        for (String setting :
                new String[] {
                    "{\"type\":\"FLEXLB\",\"masterAddress\":\"navi-master:17003\"}",
                    "{\"type\":\"UNKNOWN\"}",
                    "{\"type\":1}",
                    "null",
                    "{\"type\":\"NAVI_SCHED\",\"unknown\":1}",
                    "{\"type\":\"NAVI_SCHED\",\"masterAddress\":\"navi-master:17003\"}"
                }) {
            assertThrows(
                    ConfigValidationException.class,
                    () -> ConfigTestFixtures.parse("{\"prefillScheduler\":" + setting + "}"),
                    setting);
        }
    }

    @Test
    void rejectsIndependentMasterAddresses() {
        assertThrows(
                ConfigValidationException.class,
                () ->
                        ConfigTestFixtures.parse(
                                """
                {"prefillScheduler":{"type":"NAVI_SCHED","masterBizName":"master_biz",
                  "masterAddress":"127.0.0.1:17003"}}
                """));
    }
}
