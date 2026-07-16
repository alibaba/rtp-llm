package org.flexlb.dispatcher;

import org.apache.commons.logging.Log;
import org.flexlb.util.JsonUtils;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.boot.logging.DeferredLogFactory;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;
import org.springframework.util.StringUtils;

import java.util.Map;

/**
 * Lets the dispatcher be enabled purely through the {@code DISPATCH_CONFIG} JSON, matching how
 * {@code FLEXLB_CONFIG} / {@code MODEL_SERVICE_CONFIG} are configured. The dispatcher beans are
 * gated on the flat property {@code dispatch.fe-pool-service-id} (relaxed-bound from the
 * {@code DISPATCH_FE_POOL_SERVICE_ID} env). An operator who configures everything through
 * {@code DISPATCH_CONFIG} would otherwise get a silently-disabled dispatcher; this expands the
 * JSON's {@code fePoolServiceId} into that property at startup so either style enables it.
 *
 * <p>Runs only when the property is not already set, so the explicit env var always wins and
 * {@link DispatcherConfiguration} stays the single source of truth on config <em>content</em>.
 */
public class DispatchConfigEnvironmentPostProcessor implements EnvironmentPostProcessor {

    static final String ENABLE_PROPERTY = "dispatch.fe-pool-service-id";

    /**
     * Post-processors run before logging is initialized, so this is a {@link DeferredLogFactory}
     * log that replays into the startup log once logging comes up — a malformed
     * {@code DISPATCH_CONFIG} must not be a completely silent no-op.
     */
    private final Log log;

    public DispatchConfigEnvironmentPostProcessor(DeferredLogFactory logFactory) {
        this.log = logFactory.getLog(DispatchConfigEnvironmentPostProcessor.class);
    }

    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        if (StringUtils.hasText(environment.getProperty(ENABLE_PROPERTY))) {
            return;
        }
        String json = environment.getProperty("DISPATCH_CONFIG");
        if (!StringUtils.hasText(json)) {
            return;
        }
        // Malformed DISPATCH_CONFIG parses to null here, leaving the dispatcher disabled rather than
        // crashing boot; if the env-var path is used instead, DispatcherConfiguration.validate reports it.
        DispatchConfig cfg = JsonUtils.toObjectOrNull(json, DispatchConfig.class);
        if (cfg == null) {
            log.warn("DISPATCH_CONFIG is set but is not valid JSON — dispatcher stays DISABLED. "
                    + "Fix the JSON or set DISPATCH_FE_POOL_SERVICE_ID to fail fast at boot instead.");
            return;
        }
        String fePoolServiceId = cfg.getFePoolServiceId();
        if (!StringUtils.hasText(fePoolServiceId)) {
            return;
        }
        environment.getPropertySources().addLast(new MapPropertySource(
                "dispatchConfigExpansion", Map.of(ENABLE_PROPERTY, fePoolServiceId)));
    }
}
