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

    @Test
    void whale_master_config_wrapper_is_unpacked_when_flexlb_config_absent() {
        // Whale/Hippo deploy templates nest FLEXLB_CONFIG inside
        // WHALE_MASTER_CONFIG.zone_process_setting.process_info.envs as a
        // [name, value] tuple. When the platform forgets to expand it into a
        // real env var, V1 silently fell back to yaml defaults (V0 router) and
        // every frontend request landed on a missing response registry. Extract
        // the inner JSON so the deploy intent is preserved.
        String inner = "{\"dpBalanceEnabled\":true,\"maxQueueSize\":42}";
        String wrapper = "{\"zone_process_setting\":{\"process_info\":{\"envs\":["
                + "[\"OTHER_VAR\",\"x\"],"
                + "[\"FLEXLB_CONFIG\"," + quote(inner) + "],"
                + "[\"YET_ANOTHER\",\"y\"]"
                + "]}}}";
        StandardEnvironment env = envWith(Map.of("WHALE_MASTER_CONFIG", wrapper));

        processor.postProcessEnvironment(env, new SpringApplication());

        assertEquals("true", env.getProperty("flexlb.dpBalanceEnabled"));
        assertEquals("42", env.getProperty("flexlb.maxQueueSize"));
    }

    @Test
    void direct_flexlb_config_wins_over_whale_wrapper() {
        // If both are set the direct env wins so an operator can override the
        // platform-provided wrapper without redeploying the whole template.
        Map<String, Object> baseline = new HashMap<>();
        baseline.put("FLEXLB_CONFIG", "{\"dpBalanceEnabled\":true}");
        baseline.put("WHALE_MASTER_CONFIG", "{\"zone_process_setting\":{\"process_info\":{\"envs\":["
                + "[\"FLEXLB_CONFIG\",\"{\\\"dpBalanceEnabled\\\":false}\"]"
                + "]}}}");
        StandardEnvironment env = envWith(baseline);

        processor.postProcessEnvironment(env, new SpringApplication());

        assertEquals("true", env.getProperty("flexlb.dpBalanceEnabled"));
    }

    @Test
    void whale_wrapper_without_flexlb_config_entry_is_a_no_op() {
        String wrapper = "{\"zone_process_setting\":{\"process_info\":{\"envs\":["
                + "[\"SOME_OTHER_VAR\",\"abc\"]"
                + "]}}}";
        StandardEnvironment env = envWith(Map.of("WHALE_MASTER_CONFIG", wrapper));

        processor.postProcessEnvironment(env, new SpringApplication());

        assertFalse(env.getPropertySources().contains(
                FlexlbConfigJsonEnvironmentPostProcessor.PROPERTY_SOURCE_NAME));
    }

    @Test
    void invalid_whale_wrapper_fails_fast() {
        StandardEnvironment env = envWith(Map.of("WHALE_MASTER_CONFIG", "{not-json}"));

        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> processor.postProcessEnvironment(env, new SpringApplication()));
        assertEquals("WHALE_MASTER_CONFIG env var is not valid JSON", ex.getMessage());
    }

    private static String quote(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '"' || c == '\\') {
                sb.append('\\');
            }
            sb.append(c);
        }
        return sb.append('"').toString();
    }
}
