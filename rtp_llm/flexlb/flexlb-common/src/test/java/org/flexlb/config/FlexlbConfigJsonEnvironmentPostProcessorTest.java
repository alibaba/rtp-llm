package org.flexlb.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.SpringApplication;
import org.springframework.core.env.MapPropertySource;
import org.springframework.core.env.StandardEnvironment;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FlexlbConfigJsonEnvironmentPostProcessorTest {

    private final FlexlbConfigJsonEnvironmentPostProcessor processor =
            new FlexlbConfigJsonEnvironmentPostProcessor();

    private static StandardEnvironment envWith(Map<String, Object> props) {
        StandardEnvironment env = new StandardEnvironment();
        env.getPropertySources().addFirst(new MapPropertySource("test", props));
        return env;
    }

    @Test
    void absent_env_var_is_a_no_op() {
        StandardEnvironment env = new StandardEnvironment();

        processor.postProcessEnvironment(env, new SpringApplication());

        assertFalse(env.getPropertySources().contains(
                FlexlbConfigJsonEnvironmentPostProcessor.PROPERTY_SOURCE_NAME));
        assertNull(env.getProperty("flexlb.enableQueueing"));
    }

    @Test
    void blank_env_var_is_a_no_op() {
        StandardEnvironment env = envWith(Map.of("FLEXLB_CONFIG", "   "));

        processor.postProcessEnvironment(env, new SpringApplication());

        assertFalse(env.getPropertySources().contains(
                FlexlbConfigJsonEnvironmentPostProcessor.PROPERTY_SOURCE_NAME));
    }

    @Test
    void json_env_is_flattened_to_flexlb_prefix() {
        // The legacy FLEXLB_CONFIG payload uses camelCase field names. Spring
        // relaxed binding turns flexlb.enableQueueing -> enableQueueing, so
        // we keep the camelCase keys as-is when flattening.
        String json = "{\"enableQueueing\":true,"
                + "\"maxQueueSize\":7,"
                + "\"dpBalanceEnabled\":true,"
                + "\"dpAssignStrategy\":\"RR\"}";
        StandardEnvironment env = envWith(Map.of("FLEXLB_CONFIG", json));

        processor.postProcessEnvironment(env, new SpringApplication());

        assertEquals("true", env.getProperty("flexlb.enableQueueing"));
        assertEquals("7", env.getProperty("flexlb.maxQueueSize"));
        assertEquals("true", env.getProperty("flexlb.dpBalanceEnabled"));
        assertEquals("RR", env.getProperty("flexlb.dpAssignStrategy"));
    }

    @Test
    void invalid_json_fails_fast() {
        // Surface a typo at startup rather than silently losing config and
        // letting routing run on stale defaults.
        StandardEnvironment env = envWith(Map.of("FLEXLB_CONFIG", "{not-json}"));

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> processor.postProcessEnvironment(env, new SpringApplication()));
        assertEquals("FLEXLB_CONFIG env var is not valid JSON", ex.getMessage());
    }

    @Test
    void post_processor_property_source_takes_precedence_over_yml() {
        // Simulate application.yml provided baseline + FLEXLB_CONFIG override.
        // The post-processor inserts at HIGHEST_PRECEDENCE / addFirst so JSON
        // env wins, matching the legacy contract where deploy scripts could
        // shadow yaml defaults without code changes.
        Map<String, Object> baseline = new HashMap<>();
        baseline.put("FLEXLB_CONFIG", "{\"enableQueueing\":true}");
        baseline.put("flexlb.enableQueueing", "false");
        StandardEnvironment env = envWith(baseline);

        processor.postProcessEnvironment(env, new SpringApplication());

        assertEquals("true", env.getProperty("flexlb.enableQueueing"));
    }
}
