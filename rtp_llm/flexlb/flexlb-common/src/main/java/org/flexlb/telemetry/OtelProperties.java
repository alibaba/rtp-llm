package org.flexlb.telemetry;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.sdk.trace.SpanLimits;
import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;

import java.util.ArrayList;
import java.util.List;

@Data
@ConfigurationProperties(prefix = "trace.otel.config")
public class OtelProperties {

    private String serviceName;
    /**
     * Instrumentation name to be used to find a Tracer.
     */
    private String instrumentationName = "trace";

    /**
     * Instrumentation version to be used to find a Tracer.
     */
    private String instrumentationVersion;

    /**
     * Sets the global default {@code Sampler} value.
     */
    private double traceIdRatioBased = 1;

    /**
     * Returns the global default max number of attributes per {@link Span}.
     */
    private int maxAttrs = SpanLimits.getDefault().getMaxNumberOfAttributes();

    /**
     * Returns the global default max number of events per {@link Span}.
     */
    private int maxEvents = SpanLimits.getDefault().getMaxNumberOfEvents();

    /**
     * Returns the global default max number of link entries per {@link Span}.
     */
    private int maxLinks = SpanLimits.getDefault().getMaxNumberOfLinks();

    /**
     * Returns the global default max number of attributes per event.
     */
    private int maxEventAttrs = SpanLimits.getDefault().getMaxNumberOfAttributesPerEvent();

    /**
     * Returns the global default max number of attributes per link.
     */
    private int maxLinkAttrs = SpanLimits.getDefault().getMaxNumberOfAttributesPerLink();

    private List<String> ignoreModules = new ArrayList<>();
}
