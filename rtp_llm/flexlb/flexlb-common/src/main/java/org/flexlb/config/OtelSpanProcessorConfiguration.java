package org.flexlb.config;

import org.flexlb.telemetry.OtelProcessorProperties;
import org.flexlb.telemetry.processor.LoggingSpanProcessor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableConfigurationProperties(OtelProcessorProperties.class)
public class OtelSpanProcessorConfiguration {

    @Bean
    @ConditionalOnProperty(value = "trace.otel.processor.logging.enabled")
    public LoggingSpanProcessor otelOtlpLoggingSpanProcessor(OtelProcessorProperties properties) {
        return LoggingSpanProcessor.create();
    }
}
