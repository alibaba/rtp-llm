package org.flexlb.interceptor;

import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerCall.Listener;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.context.Scope;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import io.opentelemetry.sdk.trace.samplers.Sampler;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.telemetry.FlexlbTrace;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class GrpcTraceInterceptorTest {

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
        sdk = OpenTelemetrySdk.builder().setTracerProvider(provider).build();
        GlobalOpenTelemetry.set(sdk);
    }

    @AfterEach
    void tearDown() {
        FlexlbTrace.configureEnabled(false);
        sdk.close();
        GlobalOpenTelemetry.resetForTest();
    }

    @Test
    void disabledTracingWithExternalProviderAndCurrentSpanOnlyPropagates() {
        FlexlbTrace.configureEnabled(false);
        Span upstream = sdk.getTracer("external").spanBuilder("owner").startSpan();
        Metadata headers = new Metadata();
        String traceparent = "00-11111111111111111111111111111111-2222222222222222-01";
        headers.put(Metadata.Key.of("traceparent", Metadata.ASCII_STRING_MARSHALLER), traceparent);
        headers.put(Metadata.Key.of("tracestate", Metadata.ASCII_STRING_MARSHALLER), "vendor=one");
        ServerCallHandler<Object, Object> next = (call, incoming) -> {
            var context = GrpcTraceInterceptor.getOtelContext();
            assertEquals(traceparent, FlexlbTrace.inject(context).get("traceparent"));
            assertEquals("vendor=one", FlexlbTrace.inject(context).get("tracestate"));
            FlexlbTrace.markBusinessError(context, 500, "FLEXLB_INTERNAL_ERROR");
            call.close(Status.INTERNAL, new Metadata());
            return new Listener<>() {};
        };
        try {
            interceptAndComplete(new GrpcTraceInterceptor(), FlexlbServiceGrpc.getScheduleMethod(), headers, next);
            try (Scope ignored = upstream.makeCurrent()) {
                interceptAndComplete(new GrpcTraceInterceptor(), FlexlbServiceGrpc.getScheduleMethod(), headers, next);
                Terminals t = intercept();
                FlexlbTrace.markBusinessError(t.handlerContext(), 500, "FLEXLB_INTERNAL_ERROR");
                t.handlerCall().close(Status.INTERNAL, new Metadata());
                t.listener().onCancel();
                t.listener().onComplete();
            }
            assertTrue(upstream.isRecording());
            assertTrue(exporter.spans.isEmpty());
        } finally {
            upstream.end();
        }
        assertEquals(1, exporter.spans.size());
        assertEquals(io.opentelemetry.api.trace.StatusCode.UNSET, exporter.spans.get(0).getStatus().getStatusCode());
        assertTrue(exporter.spans.get(0).getAttributes().isEmpty());
    }

    @Test
    void namesEachRpcFromItsMethodAndPreservesRemoteParent() {
        Metadata headers = new Metadata();
        headers.put(Metadata.Key.of("traceparent", Metadata.ASCII_STRING_MARSHALLER),
                "00-11111111111111111111111111111111-2222222222222222-01");
        ServerInterceptor interceptor = new GrpcTraceInterceptor();
        ServerCallHandler<Object, Object> next =
                (call, requestHeaders) -> new Listener<>() {};

        interceptAndComplete(interceptor, FlexlbServiceGrpc.getScheduleMethod(), headers, next);
        interceptAndComplete(interceptor, FlexlbServiceGrpc.getGetRequestStateMethod(), headers, next);

        Map<String, SpanData> spans = exporter.spans.stream()
                .collect(Collectors.toMap(SpanData::getName, span -> span));
        assertEquals(2, spans.size());
        assertTrue(spans.containsKey("rtp_llm.flexlb.schedule"));
        assertTrue(spans.containsKey("rtp_llm.flexlb.get_request_state"));
        for (SpanData span : spans.values()) {
            assertEquals(SpanKind.SERVER, span.getKind());
            assertEquals("11111111111111111111111111111111", span.getTraceId().toString());
            assertEquals("2222222222222222", span.getParentSpanContext().getSpanId().toString());
        }
    }

    @Test
    void expectedBusinessErrorIsErrorWithoutExceptionEvent() {
        Metadata headers = new Metadata();
        AtomicReference<io.opentelemetry.context.Context> capturedContext = new AtomicReference<>();
        ServerCallHandler<Object, Object> next =
                (call, requestHeaders) -> {
                    capturedContext.set(GrpcTraceInterceptor.getOtelContext());
                    return new Listener<>() {};
                };
        @SuppressWarnings("unchecked")
        ServerCall<Object, Object> call = mock(ServerCall.class);
        @SuppressWarnings({"rawtypes", "unchecked"})
        MethodDescriptor<Object, Object> scheduleMethod =
                (MethodDescriptor) FlexlbServiceGrpc.getScheduleMethod();
        when(call.getMethodDescriptor()).thenReturn(
                scheduleMethod);
        ServerCall.Listener<Object> listener =
                new GrpcTraceInterceptor().interceptCall(call, headers, next);

        org.flexlb.telemetry.FlexlbTrace.markBusinessError(
                capturedContext.get(), 8402, "FLEXLB_BUSINESS_REJECTED");
        listener.onComplete();

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals(io.opentelemetry.api.trace.StatusCode.ERROR, span.getStatus().getStatusCode());
        assertEquals("FLEXLB_BUSINESS_REJECTED",
                span.getAttributes().get(io.opentelemetry.api.common.AttributeKey.stringKey("error.type")));
        assertEquals(8402L,
                span.getAttributes().get(io.opentelemetry.api.common.AttributeKey.longKey("flexlb.schedule.code")));
        assertTrue(span.getEvents().isEmpty());
    }

    @Test
    void preexistingServerSpanStillCarriesBusinessError() {
        // Synthetic upstream-owner test: create an SDK SERVER span and make it
        // current to stand in for whatever might open one upstream (an
        // auto-instrumentation agent, or another interceptor) -- no real
        // -javaagent is loaded here. The interceptor must take the existing-span
        // branch, keep ownsSpan=false, and neither end that span nor lose the
        // business rejection recorded while its owner still holds it.
        Span upstreamSpan = GlobalOpenTelemetry.getTracer("upstream")
                .spanBuilder("rtp_llm.flexlb.schedule")
                .setSpanKind(SpanKind.SERVER)
                .startSpan();

        AtomicReference<io.opentelemetry.context.Context> capturedContext = new AtomicReference<>();
        ServerCallHandler<Object, Object> next =
                (call, requestHeaders) -> {
                    capturedContext.set(GrpcTraceInterceptor.getOtelContext());
                    return new Listener<>() {};
                };
        @SuppressWarnings("unchecked")
        ServerCall<Object, Object> call = mock(ServerCall.class);
        @SuppressWarnings({"rawtypes", "unchecked"})
        MethodDescriptor<Object, Object> scheduleMethod =
                (MethodDescriptor) FlexlbServiceGrpc.getScheduleMethod();
        when(call.getMethodDescriptor()).thenReturn(scheduleMethod);

        try (Scope ignored = upstreamSpan.makeCurrent()) {
            ServerCall.Listener<Object> listener =
                    new GrpcTraceInterceptor().interceptCall(call, new Metadata(), next);
            org.flexlb.telemetry.FlexlbTrace.markBusinessError(
                    capturedContext.get(), 8402, "FLEXLB_BUSINESS_REJECTED");
            listener.onComplete();
            // The pre-existing upstream span owns the lifecycle; the interceptor
            // must not end it.
            assertEquals(0, exporter.spans.size());
        }
        upstreamSpan.end();

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals(io.opentelemetry.api.trace.StatusCode.ERROR, span.getStatus().getStatusCode());
        assertEquals("FLEXLB_BUSINESS_REJECTED",
                span.getAttributes().get(io.opentelemetry.api.common.AttributeKey.stringKey("error.type")));
        assertEquals(8402L,
                span.getAttributes().get(io.opentelemetry.api.common.AttributeKey.longKey("flexlb.schedule.code")));
        assertTrue(span.getEvents().isEmpty());
    }

    @Test
    void noopProviderStillPropagatesRemoteTraceparentDownstream() {
        // Regression guard for the default FlexLB startup (no OTel provider):
        // GlobalOpenTelemetry is a no-op so startServer() makes no real span, but
        // the no-op SpanBuilder wraps the extracted parent's SpanContext, so the
        // incoming traceparent must survive into the published context and be
        // re-injected byte-for-byte for the downstream forward. An invalid span
        // must never clobber the remote parent.
        GlobalOpenTelemetry.resetForTest();
        try {
            Metadata headers = new Metadata();
            headers.put(Metadata.Key.of("traceparent", Metadata.ASCII_STRING_MARSHALLER),
                    "00-11111111111111111111111111111111-2222222222222222-01");
            AtomicReference<io.opentelemetry.context.Context> capturedContext = new AtomicReference<>();
            ServerCallHandler<Object, Object> next =
                    (call, requestHeaders) -> {
                        capturedContext.set(GrpcTraceInterceptor.getOtelContext());
                        return new Listener<>() {};
                    };
            @SuppressWarnings("unchecked")
            ServerCall<Object, Object> call = mock(ServerCall.class);
            @SuppressWarnings({"rawtypes", "unchecked"})
            MethodDescriptor<Object, Object> scheduleMethod =
                    (MethodDescriptor) FlexlbServiceGrpc.getScheduleMethod();
            when(call.getMethodDescriptor()).thenReturn(scheduleMethod);

            ServerCall.Listener<Object> listener =
                    new GrpcTraceInterceptor().interceptCall(call, headers, next);
            listener.onComplete();

            // No provider -> no span exported, by design (fail-open).
            assertEquals(0, exporter.spans.size());
            // ...but the downstream carrier still carries the original trace/span.
            Map<String, String> carrier =
                    org.flexlb.telemetry.FlexlbTrace.inject(capturedContext.get());
            String traceparent = carrier.get("traceparent");
            assertTrue(traceparent != null
                            && traceparent.contains("11111111111111111111111111111111")
                            && traceparent.contains("2222222222222222"),
                    "downstream traceparent must preserve the remote trace/span: " + traceparent);
        } finally {
            // Restore the SDK provider so tearDown()'s sdk.close() stays valid.
            GlobalOpenTelemetry.resetForTest();
            GlobalOpenTelemetry.set(sdk);
        }
    }

    /**
     * close(OK) is the ordinary success path: the canonical status lands on the
     * span so a dashboard can filter on it, and the span stays OK.
     */
    @Test
    void okCloseRecordsCanonicalStatusAndKeepsSpanOk() {
        Terminals t = intercept();
        t.handlerCall().close(Status.OK, new Metadata());
        t.listener().onComplete();

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals(io.opentelemetry.api.trace.StatusCode.OK, span.getStatus().getStatusCode());
        assertEquals("OK", stringAttr(span, "rpc.response.status_code"));
        assertEquals(0L, longAttr(span, "rpc.grpc.status_code"));
        assertNull(stringAttr(span, "error.type"));
    }

    /**
     * The regression this wrapping exists for: onComplete() fires for an error
     * close exactly as it does for a successful one, so before close() was
     * observed the span reported OK and the real cause was lost.
     */
    @Test
    void nonOkCloseMarksSpanErrorWithCanonicalStatus() {
        Terminals t = intercept();
        t.handlerCall().close(Status.INTERNAL.withDescription("boom"), new Metadata());
        t.listener().onComplete();

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals(io.opentelemetry.api.trace.StatusCode.ERROR, span.getStatus().getStatusCode());
        assertNotEquals(io.opentelemetry.api.trace.StatusCode.OK, span.getStatus().getStatusCode());
        assertEquals("INTERNAL", stringAttr(span, "rpc.response.status_code"));
        assertEquals("INTERNAL", stringAttr(span, "error.type"));
        assertEquals((long) Status.Code.INTERNAL.value(), longAttr(span, "rpc.grpc.status_code"));
        // The description is the canonical code, never the handler's raw message,
        // which may carry endpoints or request data.
        assertEquals("INTERNAL", span.getStatus().getDescription());
    }

    /** No close observed: the synthetic cancellation is the documented fallback. */
    @Test
    void cancelWithoutCloseFallsBackToCancellation() {
        Terminals t = intercept();
        t.listener().onCancel();
        t.listener().onCancel();
        t.listener().onComplete();

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals(io.opentelemetry.api.trace.StatusCode.ERROR, span.getStatus().getStatusCode());
        assertEquals("CANCELLED", stringAttr(span, "rpc.response.status_code"));
        assertEquals("CANCELLED", stringAttr(span, "error.type"));
        assertEquals(1L, longAttr(span, "rpc.grpc.status_code"));
        assertEquals(1L, longAttr(span, "rtp_llm.grpc_status_code"));
        assertTrue(span.getEvents().isEmpty());
    }

    /**
     * A close the handler already performed outranks the synthetic cancellation:
     * onCancel() is only a fallback for the case where no close was seen.
     */
    @Test
    void cancelAfterCloseKeepsTheObservedStatus() {
        Terminals t = intercept();
        t.handlerCall().close(Status.UNAVAILABLE, new Metadata());
        t.listener().onCancel();

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals("UNAVAILABLE", stringAttr(span, "rpc.response.status_code"));
        assertEquals("UNAVAILABLE", stringAttr(span, "error.type"));
    }

    /** Repeated notifications must not export the span twice. */
    @Test
    void repeatedTerminalNotificationsEndTheSpanOnce() {
        Terminals t = intercept();
        t.handlerCall().close(Status.OK, new Metadata());
        t.handlerCall().close(Status.INTERNAL, new Metadata());
        t.listener().onComplete();
        t.listener().onComplete();
        t.listener().onCancel();

        assertEquals(1, exporter.spans.size());
        // First close wins, matching gRPC's own rejection of a second close.
        assertEquals("OK", stringAttr(exporter.spans.get(0), "rpc.response.status_code"));
    }

    /**
     * A business rejection names the cause more precisely than the transport code,
     * so it keeps the status while the transport code is still recorded.
     */
    @Test
    void businessErrorOutranksNonOkCloseStatus() {
        Terminals t = intercept();
        org.flexlb.telemetry.FlexlbTrace.markBusinessError(
                t.handlerContext(), 8402, "FLEXLB_BUSINESS_REJECTED");
        t.handlerCall().close(Status.INTERNAL, new Metadata());
        t.listener().onComplete();

        assertEquals(1, exporter.spans.size());
        SpanData span = exporter.spans.get(0);
        assertEquals(io.opentelemetry.api.trace.StatusCode.ERROR, span.getStatus().getStatusCode());
        assertEquals("FLEXLB_BUSINESS_REJECTED", stringAttr(span, "error.type"));
        assertEquals(8402L, longAttr(span, "flexlb.schedule.code"));
        assertEquals("INTERNAL", stringAttr(span, "rpc.response.status_code"));
    }

    private static void interceptAndComplete(ServerInterceptor interceptor,
                                              MethodDescriptor<?, ?> method,
                                              Metadata headers,
                                              ServerCallHandler<Object, Object> next) {
        @SuppressWarnings("unchecked")
        ServerCall<Object, Object> call = mock(ServerCall.class);
        when(call.getMethodDescriptor()).thenReturn((MethodDescriptor<Object, Object>) method);
        @SuppressWarnings("unchecked")
        ServerCall.Listener<Object> listener = interceptor.interceptCall(call, headers, next);
        listener.onComplete();
    }

    /**
     * Drives one intercepted call, handing the test the call the handler sees (so
     * it can close it the way a real handler does) and the listener gRPC would
     * notify.
     */
    private Terminals intercept() {
        AtomicReference<ServerCall<Object, Object>> handlerCall = new AtomicReference<>();
        AtomicReference<io.opentelemetry.context.Context> handlerContext = new AtomicReference<>();
        ServerCallHandler<Object, Object> next = (call, requestHeaders) -> {
            handlerCall.set(call);
            handlerContext.set(GrpcTraceInterceptor.getOtelContext());
            return new Listener<>() {};
        };
        @SuppressWarnings("unchecked")
        ServerCall<Object, Object> call = mock(ServerCall.class);
        @SuppressWarnings({"rawtypes", "unchecked"})
        MethodDescriptor<Object, Object> scheduleMethod =
                (MethodDescriptor) FlexlbServiceGrpc.getScheduleMethod();
        when(call.getMethodDescriptor()).thenReturn(scheduleMethod);
        ServerCall.Listener<Object> listener =
                new GrpcTraceInterceptor().interceptCall(call, new Metadata(), next);
        return new Terminals(listener, handlerCall.get(), handlerContext.get());
    }

    private record Terminals(ServerCall.Listener<Object> listener,
                             ServerCall<Object, Object> handlerCall,
                             io.opentelemetry.context.Context handlerContext) {
    }

    private static String stringAttr(SpanData span, String key) {
        return span.getAttributes().get(io.opentelemetry.api.common.AttributeKey.stringKey(key));
    }

    private static Long longAttr(SpanData span, String key) {
        return span.getAttributes().get(io.opentelemetry.api.common.AttributeKey.longKey(key));
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
