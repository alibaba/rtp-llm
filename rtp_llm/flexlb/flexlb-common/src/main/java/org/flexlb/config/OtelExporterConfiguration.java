package org.flexlb.config;

import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporter;
import io.opentelemetry.exporter.otlp.trace.OtlpGrpcSpanExporterBuilder;
import org.flexlb.telemetry.OtelExporterProperties;
import org.flexlb.telemetry.exporter.LoggingSpanExporter;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;

import java.util.Map;
import java.util.concurrent.TimeUnit;

@Configuration
@EnableConfigurationProperties(OtelExporterProperties.class)
public class OtelExporterConfiguration {

    @Bean
    @ConditionalOnProperty(value = "trace.otel.exporter.otlp.enabled", matchIfMissing = true)
    OtlpGrpcSpanExporter otelOtlpGrpcSpanExporter(OtelExporterProperties properties) {
        OtlpGrpcSpanExporterBuilder builder = OtlpGrpcSpanExporter.builder();
        String endpoint = properties.getOtlp().getEndpoint();
        if (StringUtils.hasText(endpoint)) {
            builder.setEndpoint(endpoint);
        }
        Long timeout = properties.getOtlp().getTimeout();
        if (timeout != null) {
            builder.setTimeout(timeout, TimeUnit.MILLISECONDS);
        }
        Map<String, String> headers = properties.getOtlp().getHeaders();
        if (!headers.isEmpty()) {
            headers.forEach(builder::addHeader);
        }
        return builder.build();
    }

    @Bean
    @ConditionalOnProperty(value = "trace.otel.exporter.logging.enabled", matchIfMissing = true)
    LoggingSpanExporter otelOtlpLoggingSpanExporter(OtelExporterProperties properties) {
        return LoggingSpanExporter.create();
    }
}