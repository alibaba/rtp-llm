package org.flexlb.telemetry.autoconfig;

import io.opentelemetry.sdk.trace.SpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;

public interface SpanProcessorProvider {

    /**
     * @param spanExporter span exporter
     * @return converted span processor
     */
    SpanProcessor toSpanProcessor(SpanExporter spanExporter);
}
