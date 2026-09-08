package org.flexlb.telemetry;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.AttributeKey;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.TraceFlags;
import io.opentelemetry.api.trace.TraceState;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.propagation.ContextPropagators;
import io.opentelemetry.context.propagation.TextMapGetter;
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
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbTraceTest {

    private RecordingExporter exporter;
    private OpenTelemetrySdk sdk;

    @BeforeEach
    void setUp() {
        FlexlbTrace.configureEnabled(true);
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
        FlexlbTrace.configureEnabled(false);
        sdk.close();
        GlobalOpenTelemetry.resetForTest();
    }

    @Test
    void disabledManualTracingPreservesPropagationWithoutTouchingExternalSpan() {
        FlexlbTrace.configureEnabled(false);
        Span owner = sdk.getTracer("external").spanBuilder("owner").startSpan();
        try {
            Context context = Context.root().with(owner);
            assertNull(FlexlbTrace.startServer("server", context));
            assertNull(FlexlbTrace.startClient("client", context));
            assertNull(FlexlbTrace.startInternal("internal", context));
            assertEquals(context, FlexlbTrace.withSpan(Span.getInvalid(), context));
            FlexlbTrace.setRequestAttributes(owner, 42L);
            FlexlbTrace.setScheduleAttribute(context, "mode", "BATCH");
            FlexlbTrace.setScheduleAttribute(context, "batch", true);
            FlexlbTrace.markBusinessError(context, 500, "FLEXLB_INTERNAL_ERROR");
            FlexlbTrace.finish(owner, new IllegalStateException("ignored"));
            FlexlbTrace.finishWithGrpcStatus(owner, "INTERNAL", 13, false);
            assertTrue(owner.isRecording());
            assertTrue(exporter.spans.isEmpty());
            Context parent = Context.root().with(Span.wrap(SpanContext.createFromRemoteParent(
                    "11111111111111111111111111111111", "2222222222222222", TraceFlags.getSampled(),
                    TraceState.builder().put("vendor", "one").build())));
            Map<String, String> carrier = FlexlbTrace.inject(parent);
            assertEquals("vendor=one", carrier.get("tracestate"));
            assertEquals(carrier, FlexlbTrace.inject(FlexlbTrace.extract(Context.root(), carrier, new MapGetter())));
        } finally {
            owner.end();
        }
        assertEquals(1, exporter.spans.size());
        assertEquals(StatusCode.UNSET, exporter.spans.get(0).getStatus().getStatusCode());
        assertTrue(exporter.spans.get(0).getAttributes().isEmpty());
        assertTrue(exporter.spans.get(0).getEvents().isEmpty());
    }

    @Test
    void scheduleFailureClassificationDoesNotGuessFromUnknownCodes() {
        assertEquals("FLEXLB_INTERNAL_ERROR", FlexlbTrace.scheduleFailureType(500));
        for (int code : new int[] {8406, 8502, 8514, 8430, 8431, 8432}) {
            assertEquals("FLEXLB_BUSINESS_REJECTED", FlexlbTrace.scheduleFailureType(code));
        }
        for (int code : new int[] {8402, 8511, 8513, 8202, -1}) {
            assertEquals("FLEXLB_SCHEDULE_FAILED", FlexlbTrace.scheduleFailureType(code));
        }
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

    /**
     * Cross-language contract: the platform indexes the unprefixed string key for
     * span search. Preserve the exact internal ID without a numeric companion.
     */
    @Test
    void requestIdIsExportedAsExactStringWithoutNumericCompanion() {
        Tracer tracer = GlobalOpenTelemetry.getTracer("test");
        Span root = tracer.spanBuilder("root").setSpanKind(SpanKind.SERVER).startSpan();
        try {
            FlexlbTrace.setRequestAttributes(root, 3540218608800727041L);
        } finally {
            root.end();
        }

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals("request_id", FlexlbTrace.REQUEST_ID);
        assertEquals("3540218608800727041",
                span.getAttributes().get(AttributeKey.stringKey(FlexlbTrace.REQUEST_ID)));
        assertFalse(span.getAttributes().asMap().keySet().stream()
                .anyMatch(key -> key.getKey().equals("rtp_llm.request_id")));
    }

    /**
     * Trace failures may only affect observation. A Span implementation that
     * throws from every method stands in for a broken provider or exporter: the
     * marking call must swallow it, because it runs on a scheduling callback whose
     * caller owns the schedule response and the routing result.
     */
    @Test
    void markBusinessErrorSwallowsAThrowingSpan() {
        Context poisoned = new ThrowingSpan().storeInContext(Context.root());
        // No try/catch here on purpose: an escaping Throwable fails the test.
        FlexlbTrace.markBusinessError(poisoned, 8402, "FLEXLB_BUSINESS_REJECTED");
    }

    /** Same contract for the terminal path, including the marker-map access. */
    @Test
    void finishSwallowsAThrowingSpan() {
        FlexlbTrace.finish(new ThrowingSpan(), null);
        FlexlbTrace.finish(new ThrowingSpan(), new IllegalStateException("boom"));
        FlexlbTrace.finishWithGrpcStatus(new ThrowingSpan(), "INTERNAL", 13, false);
        FlexlbTrace.finishWithGrpcStatus(new ThrowingSpan(), "OK", 0, true);
    }

    /**
     * Throws from every member, including the identity methods the marker map's
     * WeakHashMap calls. Reports a valid SpanContext so markBusinessError() gets
     * past its validity check and actually exercises the guarded work.
     */
    private static final class ThrowingSpan implements Span {
        @Override
        public <T> Span setAttribute(AttributeKey<T> key, T value) {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public Span addEvent(String name, Attributes attributes) {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public Span addEvent(String name, Attributes attributes, long timestamp, TimeUnit unit) {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public Span setStatus(StatusCode statusCode, String description) {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public Span recordException(Throwable exception, Attributes additionalAttributes) {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public Span updateName(String name) {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public void end() {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public void end(long timestamp, TimeUnit unit) {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public SpanContext getSpanContext() {
            return SpanContext.create("11111111111111111111111111111111", "2222222222222222",
                    TraceFlags.getSampled(), TraceState.getDefault());
        }

        @Override
        public boolean isRecording() {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public int hashCode() {
            throw new IllegalStateException("span is poisoned");
        }

        @Override
        public boolean equals(Object other) {
            throw new IllegalStateException("span is poisoned");
        }
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
