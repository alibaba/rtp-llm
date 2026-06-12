package org.flexlb.dispatcher;

import org.junit.jupiter.api.Test;
import org.springframework.mock.env.MockEnvironment;

import static org.flexlb.dispatcher.DispatchConfigEnvironmentPostProcessor.ENABLE_PROPERTY;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

class DispatchConfigEnvironmentPostProcessorTest {

    private final DispatchConfigEnvironmentPostProcessor epp = new DispatchConfigEnvironmentPostProcessor();

    @Test
    void jsonOnlyConfigEnablesDispatcher() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty("DISPATCH_CONFIG", "{\"fePoolServiceId\":\"rtp_llm.frontend.service\"}");

        epp.postProcessEnvironment(env, null);

        assertEquals("rtp_llm.frontend.service", env.getProperty(ENABLE_PROPERTY),
                "DISPATCH_CONFIG.fePoolServiceId must populate the enable property on its own");
    }

    @Test
    void explicitEnablePropertyWins() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty(ENABLE_PROPERTY, "explicit.service");
        env.setProperty("DISPATCH_CONFIG", "{\"fePoolServiceId\":\"json.service\"}");

        epp.postProcessEnvironment(env, null);

        assertEquals("explicit.service", env.getProperty(ENABLE_PROPERTY),
                "an explicit DISPATCH_FE_POOL_SERVICE_ID must not be overridden by the JSON");
    }

    @Test
    void malformedConfigDoesNotEnableAndDoesNotThrow() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty("DISPATCH_CONFIG", "{not valid json");

        epp.postProcessEnvironment(env, null);

        assertNull(env.getProperty(ENABLE_PROPERTY),
                "a malformed DISPATCH_CONFIG must leave the dispatcher disabled, not crash boot");
    }

    @Test
    void blankFePoolServiceIdDoesNotEnable() {
        MockEnvironment env = new MockEnvironment();
        env.setProperty("DISPATCH_CONFIG", "{\"fePoolServiceId\":\"  \"}");

        epp.postProcessEnvironment(env, null);

        assertNull(env.getProperty(ENABLE_PROPERTY));
    }
}
