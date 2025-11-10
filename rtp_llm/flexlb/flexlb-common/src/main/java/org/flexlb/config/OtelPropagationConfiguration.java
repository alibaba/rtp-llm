package org.flexlb.config;

import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.context.propagation.TextMapPropagator;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.List;

@Configuration
public class OtelPropagationConfiguration {

    @Bean
    ContextPropagators otelContextPropagators(ObjectProvider<List<TextMapPropagator>> propagators) {
        return ContextPropagators.create(W3CTraceContextPropagator.getInstance());
    }
}
