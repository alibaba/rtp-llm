package org.flexlb.config;

import com.google.common.collect.Lists;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.OpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.resources.Resource;
import io.opentelemetry.sdk.trace.*;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessor;
import io.opentelemetry.sdk.trace.export.BatchSpanProcessorBuilder;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import io.opentelemetry.sdk.trace.samplers.Sampler;
import org.apache.commons.collections4.ListUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.telemetry.OtelExporterProperties;
import org.flexlb.telemetry.OtelProcessorProperties;
import org.flexlb.telemetry.OtelProperties;
import org.flexlb.telemetry.OtelResourceProperties;
import org.flexlb.telemetry.autoconfig.SpanProcessorProvider;
import org.flexlb.telemetry.extension.eagleeye.EagleEyeIdGenerator;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.core.env.Environment;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.stream.Collectors;

@Configuration
@ConditionalOnClass({Tracer.class, OtelProperties.class, OtelResourceProperties.class, ContextPropagators.class})
@EnableConfigurationProperties(value = {OtelProperties.class, OtelExporterProperties.class, OtelProcessorProperties.class, OtelResourceProperties.class})
@Order(Ordered.HIGHEST_PRECEDENCE)
public class OtelConfiguration {

    @Bean
    OpenTelemetry otel(SdkTracerProvider tracerProvider, ContextPropagators contextPropagators) {
        GlobalOpenTelemetry.resetForTest();
        return OpenTelemetrySdk.builder().setTracerProvider(tracerProvider).setPropagators(contextPropagators).buildAndRegisterGlobal();
    }

    @Bean
    SdkTracerProvider otelTracerProvider(SpanLimits spanLimits, ObjectProvider<List<SpanProcessor>> spanProcessors, ObjectProvider<List<SpanExporter>> spanExporters, Sampler sampler, Resource resource, IdGenerator idGenerator, SpanProcessorProvider spanProcessorProvider) {
        SdkTracerProviderBuilder sdkTracerProviderBuilder = SdkTracerProvider.builder().setResource(resource).setSampler(sampler).setSpanLimits(spanLimits).setIdGenerator(idGenerator);
        List<SpanProcessor> processors = ListUtils.defaultIfNull(spanProcessors.getIfAvailable(), Lists.newArrayList());
        processors.addAll(ListUtils.defaultIfNull(spanExporters.getIfAvailable(), Lists.newArrayList()).stream().map(spanProcessorProvider::toSpanProcessor).collect(Collectors.toList()));
        processors.forEach(sdkTracerProviderBuilder::addSpanProcessor);
        return sdkTracerProviderBuilder.build();
    }

    @Bean
    IdGenerator otelIdGenerator() {
        return new EagleEyeIdGenerator();
    }

    @Bean
    SpanProcessorProvider otelBatchSpanProcessorProvider(OtelProcessorProperties otelProcessorProperties) {
        return new SpanProcessorProvider() {
            @Override
            public SpanProcessor toSpanProcessor(SpanExporter spanExporter) {
                BatchSpanProcessorBuilder builder = BatchSpanProcessor.builder(spanExporter);
                setBuilderProperties(otelProcessorProperties, builder);
                return builder.build();
            }

            private void setBuilderProperties(OtelProcessorProperties otelProcessorProperties, BatchSpanProcessorBuilder builder) {
                if (otelProcessorProperties.getBatch().getExporterTimeout() != null) {
                    builder.setExporterTimeout(otelProcessorProperties.getBatch().getExporterTimeout(), TimeUnit.MILLISECONDS);
                }
                if (otelProcessorProperties.getBatch().getMaxExportBatchSize() != null) {
                    builder.setMaxExportBatchSize(otelProcessorProperties.getBatch().getMaxExportBatchSize());
                }
                if (otelProcessorProperties.getBatch().getMaxQueueSize() != null) {
                    builder.setMaxQueueSize(otelProcessorProperties.getBatch().getMaxQueueSize());
                }
                if (otelProcessorProperties.getBatch().getScheduleDelay() != null) {
                    builder.setScheduleDelay(otelProcessorProperties.getBatch().getScheduleDelay(), TimeUnit.MILLISECONDS);
                }
            }
        };
    }

    @Bean
    Resource otelResource(Environment env, ObjectProvider<List<Supplier<Resource>>> resourceProviders, OtelProperties otelProperties, OtelResourceProperties otelResourceProperties) {
        String applicationName = StringUtils.defaultString(otelProperties.getServiceName(), env.getProperty("spring.application.name"));
        Resource resource = defaultResource(applicationName, otelResourceProperties);
        List<Supplier<Resource>> resourceCustomizers = ListUtils.defaultIfNull(resourceProviders.getIfAvailable(), Lists.newArrayList());
        for (Supplier<Resource> provider : resourceCustomizers) {
            resource = resource.merge(provider.get());
        }
        return resource;
    }

    private Resource defaultResource(String applicationName, OtelResourceProperties otelResourceProperties) {
        if (applicationName == null) {
            return Resource.empty();
        }
        String environment = "";
        if (StringUtils.isNotBlank(otelResourceProperties.getEnv())) {
            environment = StringUtils.trim(otelResourceProperties.getEnv());
        } else {
//            String environment = EnvironmentUtils.getEnvironment();
        }
        return Resource.empty()
                .merge(Resource.builder()
                        .put(AttributeKey.stringKey("service.name"), applicationName)
//                        .put("ip", EnvironmentUtils.getIp())
                        .put("role", System.getenv(otelResourceProperties.getRoleEnv()))
//                        .put("idc", EnvironmentUtils.getTagIdc())
                        .put("env", environment)
                        .put("platform", "whale-wave")
                        .build());
    }

    @Bean
    SpanLimits otelSpanLimits(OtelProperties otelProperties) {
        return SpanLimits.getDefault().toBuilder().setMaxNumberOfAttributes(otelProperties.getMaxAttrs()).setMaxNumberOfAttributesPerEvent(otelProperties.getMaxEventAttrs()).setMaxNumberOfAttributesPerLink(otelProperties.getMaxLinkAttrs()).setMaxNumberOfEvents(otelProperties.getMaxEvents()).setMaxNumberOfLinks(otelProperties.getMaxLinks()).build();
    }

    @Bean
    Tracer otelTracer(OpenTelemetry openTelemetry, OtelProperties otelProperties) {
        return openTelemetry.getTracer(otelProperties.getInstrumentationName());
    }

    @Bean
    Sampler otelSampler(OtelProperties otelProperties) {
        return Sampler.traceIdRatioBased(otelProperties.getTraceIdRatioBased());
    }
}
