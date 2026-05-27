package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.util.Logger;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;

import java.util.Map;

/**
 * Bridges {@code DISPATCH_CONFIG.enabled} (a JSON field carried inside an env var) to the Spring
 * property {@code dispatch.enabled} so {@code @ConditionalOnProperty} on dispatcher beans sees
 * the value during context refresh. Spring's {@code SystemEnvironmentPropertySource} maps
 * {@code DISPATCH_ENABLED} to {@code dispatch.enabled} natively, but it does NOT unpack JSON
 * carried in {@code DISPATCH_CONFIG}.
 *
 * <p>Registered through {@code META-INF/spring.factories} so it runs early in
 * {@link SpringApplication#run} — before any bean conditional is evaluated. JSON parse failures
 * MUST emit a WARN here: {@link DispatcherConfiguration} is itself gated on
 * {@code dispatch.enabled=true}, so a swallowed parse error means the {@code @Bean dispatchConfig()}
 * validator never runs either — dispatcher silently stays disabled with no other signal.
 */
public class DispatchEnvironmentPostProcessor implements EnvironmentPostProcessor {

    private static final String JSON_ENV_VAR = "DISPATCH_CONFIG";
    private static final String SPRING_PROPERTY = "dispatch.enabled";
    private static final String SOURCE_NAME = "dispatchConfigBridge";

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication app) {
        String json = environment.getProperty(JSON_ENV_VAR);
        if (json == null || json.isBlank()) {
            return;
        }
        try {
            JsonNode tree = new ObjectMapper().readTree(json);
            JsonNode enabled = tree.get("enabled");
            if (enabled != null && enabled.isBoolean()) {
                Map<String, Object> props = Map.of(SPRING_PROPERTY, enabled.booleanValue());
                environment.getPropertySources().addFirst(new MapPropertySource(SOURCE_NAME, props));
            } else if (enabled != null) {
                // {"enabled":"true"} (string) and similar type-mismatches would slip through the
                // isBoolean() check; without this WARN the operator's typo silently disables the
                // dispatcher just like a JSON syntax error would.
                Logger.warn("DISPATCH_CONFIG.enabled is not a JSON boolean (got {}); "
                        + "dispatcher will stay disabled. Use true/false without quotes.",
                        enabled.getNodeType());
            }
        } catch (Exception e) {
            Logger.warn("DISPATCH_CONFIG JSON parse failed; dispatcher will stay disabled. "
                    + "Check env var JSON syntax. err={}: {}",
                    e.getClass().getSimpleName(), e.getMessage());
        }
    }
}
