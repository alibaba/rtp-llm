package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.Ordered;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;

import java.util.HashMap;
import java.util.Map;

/**
 * Translates the legacy {@code FLEXLB_CONFIG} JSON env var into Spring's
 * Environment as {@code flexlb.<field>} properties so {@link FlexlbConfig}'s
 * {@code @ConfigurationProperties} binding can consume it alongside
 * application.yml. Runs ahead of standard property sources so a deploy script
 * setting only {@code FLEXLB_CONFIG} keeps producing the same effective config.
 *
 * <p>When {@code FLEXLB_CONFIG} is absent but the platform passes the wrapper
 * {@code WHALE_MASTER_CONFIG} (Whale/Hippo deploy templates that nest the real
 * env vars as {@code zone_process_setting.process_info.envs[[name,value]]}),
 * extract the {@code FLEXLB_CONFIG} entry from inside the wrapper instead of
 * silently falling through to yaml defaults.
 */
public class FlexlbConfigJsonEnvironmentPostProcessor implements EnvironmentPostProcessor, Ordered {

    static final String PROPERTY_SOURCE_NAME = "flexlbConfigJson";
    private static final String PROPERTY_PREFIX = "flexlb.";
    private static final String FLEXLB_CONFIG_KEY = "FLEXLB_CONFIG";
    private static final String WHALE_WRAPPER_KEY = "WHALE_MASTER_CONFIG";

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        String json = resolveFlexlbConfigJson(environment);
        if (json == null || json.isBlank()) {
            return;
        }
        Map<String, Object> flat;
        try {
            Map<?, ?> root = objectMapper.readValue(json, Map.class);
            flat = flatten(root);
        } catch (Exception e) {
            throw new IllegalStateException("FLEXLB_CONFIG env var is not valid JSON", e);
        }
        environment.getPropertySources().addFirst(new MapPropertySource(PROPERTY_SOURCE_NAME, flat));
    }

    private String resolveFlexlbConfigJson(ConfigurableEnvironment environment) {
        String direct = environment.getProperty(FLEXLB_CONFIG_KEY);
        if (direct != null && !direct.isBlank()) {
            return direct;
        }
        String wrapper = environment.getProperty(WHALE_WRAPPER_KEY);
        if (wrapper == null || wrapper.isBlank()) {
            return null;
        }
        return extractFromWhaleWrapper(wrapper);
    }

    private String extractFromWhaleWrapper(String wrapperJson) {
        JsonNode envs;
        try {
            envs = objectMapper.readTree(wrapperJson)
                    .path("zone_process_setting")
                    .path("process_info")
                    .path("envs");
        } catch (Exception e) {
            throw new IllegalStateException("WHALE_MASTER_CONFIG env var is not valid JSON", e);
        }
        if (!envs.isArray()) {
            return null;
        }
        for (JsonNode entry : envs) {
            if (entry.isArray() && entry.size() >= 2
                    && FLEXLB_CONFIG_KEY.equals(entry.get(0).asText())) {
                return entry.get(1).asText();
            }
        }
        return null;
    }

    private static Map<String, Object> flatten(Map<?, ?> root) {
        Map<String, Object> out = new HashMap<>();
        for (Map.Entry<?, ?> e : root.entrySet()) {
            out.put(PROPERTY_PREFIX + e.getKey(), e.getValue());
        }
        return out;
    }

    @Override
    public int getOrder() {
        return Ordered.HIGHEST_PRECEDENCE;
    }
}
