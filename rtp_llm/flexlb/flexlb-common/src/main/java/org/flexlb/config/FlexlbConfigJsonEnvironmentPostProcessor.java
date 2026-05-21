package org.flexlb.config;

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
 */
public class FlexlbConfigJsonEnvironmentPostProcessor implements EnvironmentPostProcessor, Ordered {

    static final String PROPERTY_SOURCE_NAME = "flexlbConfigJson";
    private static final String PROPERTY_PREFIX = "flexlb.";

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        String json = environment.getProperty("FLEXLB_CONFIG");
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
