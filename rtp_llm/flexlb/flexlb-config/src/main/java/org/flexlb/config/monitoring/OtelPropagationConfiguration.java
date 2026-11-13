package org.flexlb.config.monitoring;

import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.propagation.ContextPropagators;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Registers W3C trace-context propagation for incoming and outgoing calls.
 */
@Configuration
public class OtelPropagationConfiguration {

    @Bean
    ContextPropagators otelContextPropagators() {
        return ContextPropagators.create(W3CTraceContextPropagator.getInstance());
    }
}
