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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

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
    void successfulOrUntracedRequestDoesNotInventTraceIdentity() {
        SpanContext span = SpanContext.create("11111111111111111111111111111111",
                "2222222222222222", TraceFlags.getSampled(), TraceState.getDefault());
        assertNull(complete(request(true, Context.root().with(Span.wrap(span)))).getTraceId());
        for (Context trace : new Context[] {null, Context.root()}) {
            PvLogData pv = complete(request(false, trace));
            assertNull(pv.getTraceId());
            assertNull(pv.getSpanId());
            assertNull(pv.getTraceSampled());
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
