package org.flexlb.dispatcher;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;
import org.springframework.core.env.StandardEnvironment;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/** Exposes Dispatcher environment settings without enabling environment binding for the rest of Master. */
public class DispatchEnvironmentPostProcessor implements EnvironmentPostProcessor {
    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        Map<String, Object> properties = new HashMap<>();
        // Only the supported deployment settings enter Spring properties.
        for (String name : List.of("fe-pool-service-id", "sub-batch", "pre-assign-be", "batch-timeout-ms", "probe-path")) {
            String value = System.getenv("DISPATCH_" + name.toUpperCase(Locale.ROOT).replace('-', '_'));
            if (value != null) {
                properties.put("dispatch." + name, value);
            }
        }
        // Keep normal precedence: command line / system properties > environment > configuration files.
        environment.getPropertySources().addAfter(StandardEnvironment.SYSTEM_ENVIRONMENT_PROPERTY_SOURCE_NAME,
                new MapPropertySource("dispatchEnvironment", properties));
    }
}
