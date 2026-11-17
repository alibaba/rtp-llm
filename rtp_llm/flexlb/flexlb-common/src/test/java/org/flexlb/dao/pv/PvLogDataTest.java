package org.flexlb.dao.pv;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.TraceFlags;
import io.opentelemetry.api.trace.TraceState;
import io.opentelemetry.context.Context;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Checks request trace correlation for synchronous and asynchronous PV records.
 */
class PvLogDataTest {
    @Test
    void failedRequestCarriesExistingTraceIdentityAndSamplingDecision() {
        for (TraceFlags flags : new TraceFlags[] {TraceFlags.getSampled(), TraceFlags.getDefault()}) {
            SpanContext span = SpanContext.create("11111111111111111111111111111111",
                    "2222222222222222", flags, TraceState.getDefault());
            BalanceContext context = request(false, Context.root().with(Span.wrap(span)));
            Map<String, Object> diagnostics = Map.of("cause", "PREFILL placement unavailable", "workers", 42);
            context.setSchedulingDiagnostics(diagnostics);
            PvLogData pv = complete(context);
            assertEquals(span.getTraceId(), pv.getTraceId());
            assertEquals(span.getSpanId(), pv.getSpanId());
            assertEquals(span.isSampled(), pv.getTraceSampled());
            assertSame(diagnostics, pv.getSchedulingDiagnostics());
        }
    }

    @Test
    void successfulCompletionUsesSavedRequestTraceOnAnotherThread() throws Exception {
        try (var executor = Executors.newSingleThreadExecutor()) {
            for (TraceFlags flags : new TraceFlags[] {TraceFlags.getSampled(), TraceFlags.getDefault()}) {
                SpanContext span = SpanContext.create("11111111111111111111111111111111",
                        "2222222222222222", flags, TraceState.getDefault());
                BalanceContext context = request(true, Context.root().with(Span.wrap(span)));
                context.getRequest().setRequestId(3633136869870391858L);
                context.setSchedulingDiagnostics(Map.of("failure-only", true));
                PvLogData pv = executor.submit(() -> complete(context)).get(5, TimeUnit.SECONDS);
                assertEquals(3633136869870391858L, pv.getRequestId());
                assertEquals(span.getTraceId(), pv.getTraceId());
                assertEquals(span.getSpanId(), pv.getSpanId());
                assertEquals(span.isSampled(), pv.getTraceSampled());
                assertNull(pv.getSchedulingDiagnostics());
            }
        }
    }

    @Test
    void untracedRequestDoesNotInventTraceIdentity() {
        for (boolean success : new boolean[] {false, true}) {
            for (Context trace : new Context[] {null, Context.root()}) {
                PvLogData pv = complete(request(success, trace));
                assertNull(pv.getTraceId());
                assertNull(pv.getSpanId());
                assertNull(pv.getTraceSampled());
            }
        }
    }

    private static BalanceContext request(boolean success, Context trace) {
        BalanceContext context = new BalanceContext(new FlexlbConfig());
        context.setRequest(new Request());
        context.setSuccess(success);
        context.setTraceContext(trace);
        return context;
    }

    private static PvLogData complete(BalanceContext context) {
        return new PvLogData(context, 8431, "RESOURCE_EXHAUSTED", "LOCAL_MASTER", 0,
                "REQUEST_STATE_TIMED_OUT", "", System.currentTimeMillis());
    }
}
