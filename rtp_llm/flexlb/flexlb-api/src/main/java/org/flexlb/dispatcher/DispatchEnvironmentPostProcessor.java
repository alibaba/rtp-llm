package org.flexlb.dispatcher;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.env.EnvironmentPostProcessor;
import org.springframework.core.env.ConfigurableEnvironment;
import org.springframework.core.env.MapPropertySource;
import org.springframework.core.env.StandardEnvironment;

import java.util.HashMap;
import java.util.Locale;
import java.util.Map;

/** Exposes Dispatcher environment settings without enabling environment binding for the rest of Master. */
public class DispatchEnvironmentPostProcessor implements EnvironmentPostProcessor {
    @Override
    public void postProcessEnvironment(ConfigurableEnvironment environment, SpringApplication application) {
        Map<String, Object> properties = new HashMap<>();
        System.getenv().forEach((name, value) -> {
            // Credentials stay outside Spring properties; the removed JSON configuration is not supported.
            if (name.startsWith("DISPATCH_") && !name.equals("DISPATCH_ROUTING_TOKEN") && !name.equals("DISPATCH_CONFIG")) {
                properties.put("dispatch." + name.substring("DISPATCH_".length()).toLowerCase(Locale.ROOT).replace('_', '-'), value);
            }
        });
        // Keep normal precedence: command line / system properties > environment > configuration files.
        environment.getPropertySources().addAfter(StandardEnvironment.SYSTEM_ENVIRONMENT_PROPERTY_SOURCE_NAME,
                new MapPropertySource("dispatchEnvironment", properties));
    }
}
