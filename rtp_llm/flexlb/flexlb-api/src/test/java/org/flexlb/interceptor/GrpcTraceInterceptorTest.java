package org.flexlb.interceptor;

import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.ServerCall.Listener;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import io.opentelemetry.sdk.trace.samplers.Sampler;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class GrpcTraceInterceptorTest {

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
        sdk = OpenTelemetrySdk.builder().setTracerProvider(provider).build();
        GlobalOpenTelemetry.set(sdk);
    }

    @AfterEach
    void tearDown() {
        sdk.close();
        GlobalOpenTelemetry.resetForTest();
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
