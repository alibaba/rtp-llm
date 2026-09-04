package org.flexlb.telemetry;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import io.opentelemetry.sdk.trace.samplers.Sampler;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbTraceTest {

    private RecordingExporter exporter;
    private OpenTelemetrySdk sdk;

    @BeforeEach
    void setUp() {
        GlobalOpenTelemetry.resetForTest();
        exporter = new RecordingExporter();
        SdkTracerProvider provider = SdkTracerProvider.builder()
                .setSampler(Sampler.alwaysOn())
                .addSpanProcessor(SimpleSpanProcessor.create(exporter))
                .build();
        sdk = OpenTelemetrySdk.builder()
                .setTracerProvider(provider)
                .setPropagators(ContextPropagators.create(
                        W3CTraceContextPropagator.getInstance()))
                .build();
        GlobalOpenTelemetry.set(sdk);
    }

    @AfterEach
    void tearDown() {
        sdk.close();
        GlobalOpenTelemetry.resetForTest();
    }

    @Test
    void injectAndExtractRoundTripUsesW3cContext() {
        Tracer tracer = GlobalOpenTelemetry.getTracer("test");
        Span root = tracer.spanBuilder("root").setSpanKind(SpanKind.SERVER).startSpan();
        try {
            Context rootContext = root.storeInContext(Context.root());
            Map<String, String> carrier = FlexlbTrace.inject(rootContext);
            assertTrue(carrier.get("traceparent").startsWith("00-"));

            Context extracted = FlexlbTrace.extract(
                    Context.root(), carrier, new MapGetter());
            assertEquals(root.getSpanContext().getTraceId(),
                    Span.fromContext(extracted).getSpanContext().getTraceId());
            assertEquals(root.getSpanContext().getSpanId(),
                    Span.fromContext(extracted).getSpanContext().getSpanId());
        } finally {
            root.end();
        }
    }

    @Test
    void scheduleAttributesStayOnExistingServerSpan() {
        Tracer tracer = GlobalOpenTelemetry.getTracer("test");
        Span root = tracer.spanBuilder("root").setSpanKind(SpanKind.SERVER).startSpan();
        try {
            Context context = root.storeInContext(Context.root());
            FlexlbTrace.setScheduleAttribute(context, FlexlbTrace.SCHEDULE_MODE, "BATCH");
            FlexlbTrace.setScheduleAttribute(context, FlexlbTrace.BATCH_ID, 42L);
            FlexlbTrace.setScheduleAttribute(context, FlexlbTrace.ENQUEUED_BY_MASTER, true);
            FlexlbTrace.setScheduleDuration(context, FlexlbTrace.BATCH_WAIT_MS,
                    1_000_000L, 8_000_000L);
        } finally {
            root.end();
        }

        assertEquals(1, exporter.spans.size());
        SpanData rootData = exporter.spans.get(0);
        assertEquals(SpanKind.SERVER, rootData.getKind());
        assertNotNull(rootData.getAttributes().get(
                AttributeKey.stringKey(FlexlbTrace.SCHEDULE_MODE)));
        assertEquals("BATCH", rootData.getAttributes().get(
                AttributeKey.stringKey(FlexlbTrace.SCHEDULE_MODE)));
        assertEquals(42L, rootData.getAttributes().get(
                AttributeKey.longKey(FlexlbTrace.BATCH_ID)));
        assertEquals(true, rootData.getAttributes().get(
                AttributeKey.booleanKey(FlexlbTrace.ENQUEUED_BY_MASTER)));
        assertEquals(7L, rootData.getAttributes().get(
                AttributeKey.longKey(FlexlbTrace.BATCH_WAIT_MS)));
    }

    private static final class MapGetter implements TextMapGetter<Map<String, String>> {
        @Override
        public Iterable<String> keys(Map<String, String> carrier) {
            return carrier.keySet();
        }

        @Override
        public String get(Map<String, String> carrier, String key) {
            return carrier.get(key);
        }
    }

    private static final class RecordingExporter implements SpanExporter {
        private final List<SpanData> spans = new ArrayList<>();

        @Override
        public CompletableResultCode export(Collection<SpanData> batch) {
            spans.addAll(batch);
            return CompletableResultCode.ofSuccess();
        }

        @Override
        public CompletableResultCode flush() {
            return CompletableResultCode.ofSuccess();
        }

        @Override
        public CompletableResultCode shutdown() {
            return CompletableResultCode.ofSuccess();
        }
    }
}
