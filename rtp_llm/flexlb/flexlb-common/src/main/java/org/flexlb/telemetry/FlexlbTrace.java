package org.flexlb.telemetry;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanBuilder;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.opentelemetry.context.propagation.TextMapPropagator;
import io.opentelemetry.context.propagation.TextMapSetter;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.enums.StatusEnum;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.WeakHashMap;
import java.util.concurrent.TimeUnit;

/**
 * Small fail-open facade for the FlexLB manual tracing points.
 *
 * <p>When RTP_LLM_OTEL_TRACE_ENABLE is enabled, the FlexLB application enables
 * the SDK provider before Spring starts. An external instrumentation provider
 * may supply it instead. The manual tracing switch is independent of provider
 * ownership; when disabled, only W3C propagation remains active.</p>
 */
public final class FlexlbTrace {

    public static final String INSTRUMENTATION_NAME = "org.flexlb";
    /**
     * Platform-indexed span search key. Unitrace only indexes this unprefixed
     * string spelling, so a span carrying just the numeric twin below cannot be
     * found by request id. Mirrors kAttrRequestId / REQUEST_ID in the C++ and
     * Python registries.
     */
    public static final String REQUEST_ID = "request_id";
    /** Numeric twin, kept for internal correlation. */
    public static final String RTP_LLM_REQUEST_ID = "rtp_llm.request_id";
    public static final String BATCH_ID = "rtp_llm.batch_id";
    public static final String BATCH_SIZE = "rtp_llm.batch_size";
    public static final String PREFILL_ADDRESS = "rtp_llm.prefill_address";
    public static final String DECODE_ADDRESS = "rtp_llm.decode_address";
    public static final String DISPATCH_REASON = "rtp_llm.dispatch_reason";
    public static final String SCHEDULE_MODE = "flexlb.schedule.mode";
    public static final String SCHEDULE_CODE = "flexlb.schedule.code";
    public static final String ENQUEUED_BY_MASTER = "rtp_llm.enqueued_by_master";
    public static final String ROUTE_SUBMIT_MS = "rtp_llm.route_submit_ms";
    public static final String BATCH_WAIT_MS = "rtp_llm.batch_wait_ms";
    public static final String ENQUEUE_BATCH_MS = "rtp_llm.enqueue_batch_ms";
    public static final String ACK_TO_RESPONSE_MS = "rtp_llm.ack_to_response_ms";
    public static final String GRPC_STATUS_CODE = "rpc.grpc.status_code";
    public static final String RTP_LLM_GRPC_STATUS_CODE = "rtp_llm.grpc_status_code";
    /**
     * Canonical gRPC terminal status on a SERVER span, mirroring the C++
     * kAttrRpcResponseStatusCode contract; GRPC_STATUS_CODE above carries the
     * numeric companion.
     */
    public static final String RPC_RESPONSE_STATUS_CODE = "rpc.response.status_code";
    public static final String ERROR_TYPE = "error.type";

    private static final TextMapSetter<Map<String, String>> MAP_SETTER =
            (carrier, key, value) -> carrier.put(key, value);
    private static final TextMapPropagator W3C_PROPAGATOR =
            W3CTraceContextPropagator.getInstance();
    // Weak keys because this class does not end every span that owns a marker:
    // an upstream-owned SERVER span is finished by its own owner, so holding it
    // strongly here would retain it forever.
    private static final Map<Span, BusinessError> BUSINESS_ERRORS =
            Collections.synchronizedMap(new WeakHashMap<>());
    private static volatile boolean enabled;

    private FlexlbTrace() {
    }

    /** Configured at startup, before requests; does not initialize or own a provider. */
    public static void configureEnabled(boolean value) {
        enabled = value;
    }

    public static boolean isEnabled() {
        return enabled;
    }

    /** Only explicit admission/validation codes imply a business rejection. */
    public static String scheduleFailureType(int code) {
        if (code == StatusEnum.INTERNAL_ERROR.getCode()) {
            return "FLEXLB_INTERNAL_ERROR";
        }
        if (code == StrategyErrorType.INVALID_REQUEST.getErrorCode()
                || code == StrategyErrorType.QUEUE_FULL.getErrorCode()
                || code == StrategyErrorType.BATCH_TOKEN_CAPACITY_EXCEEDED.getErrorCode()
                || code == StrategyErrorType.PRIORITY_ADMISSION_REJECTED.getErrorCode()
                || code == StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode()
                || code == StrategyErrorType.ADMISSION_UNAVAILABLE.getErrorCode()) {
            return "FLEXLB_BUSINESS_REJECTED";
        }
        return "FLEXLB_SCHEDULE_FAILED";
    }

    public static Span startServer(String name, Context parent) {
        return start(name, parent, SpanKind.SERVER);
    }

    public static Span startInternal(String name, Context parent) {
        return start(name, parent, SpanKind.INTERNAL);
    }

    public static Span startClient(String name, Context parent) {
        return start(name, parent, SpanKind.CLIENT);
    }

    public static Context withSpan(Span span, Context fallback) {
        try {
            Context base = fallback == null ? Context.current() : fallback;
            return !enabled || span == null ? base : span.storeInContext(base);
        } catch (Throwable ignored) {
            return fallback == null ? Context.current() : fallback;
        }
    }

    public static SpanContext spanContext(Context context) {
        try {
            if (context == null) {
                return null;
            }
            SpanContext spanContext = Span.fromContext(context).getSpanContext();
            return spanContext.isValid() ? spanContext : null;
        } catch (Throwable ignored) {
            return null;
        }
    }

    public static Map<String, String> inject(Context context) {
        Map<String, String> carrier = new HashMap<>();
        try {
            Context source = context == null ? Context.current() : context;
            W3C_PROPAGATOR.inject(source, carrier, MAP_SETTER);
        } catch (Throwable ignored) {
            carrier.clear();
        }
        return carrier;
    }

    public static <C> Context extract(Context parent, C carrier, TextMapGetter<C> getter) {
        try {
            return W3C_PROPAGATOR.extract(
                    parent == null ? Context.current() : parent, carrier, getter);
        } catch (Throwable ignored) {
            return parent == null ? Context.current() : parent;
        }
    }

    public static void setRequestAttributes(Span span, long requestId) {
        // Double-write, matching the C++ and Python registries: the platform
        // indexes the string key for span search while the numeric twin stays for
        // internal correlation.
        setAttribute(span, REQUEST_ID, Long.toString(requestId));
        setAttribute(span, RTP_LLM_REQUEST_ID, requestId);
    }

    public static void setAttribute(Span span, String key, long value) {
        try {
            if (enabled && span != null) {
                span.setAttribute(key, value);
            }
        } catch (Throwable ignored) {
            // Trace must never affect routing or dispatch.
        }
    }

    public static void setAttribute(Span span, String key, String value) {
        try {
            if (enabled && span != null && value != null) {
                span.setAttribute(key, value);
            }
        } catch (Throwable ignored) {
            // Trace must never affect routing or dispatch.
        }
    }

    public static void setAttribute(Span span, String key, boolean value) {
        try {
            if (enabled && span != null) {
                span.setAttribute(key, value);
            }
        } catch (Throwable ignored) {
            // Trace must never affect routing or dispatch.
        }
    }

    /**
     * Adds an attribute to the already-created Schedule SERVER span. The
     * scheduling work is asynchronous, so this deliberately does not create
     * a child span or replace the request context.
     */
    public static void setScheduleAttribute(Context context, String key, long value) {
        setAttribute(spanFromContext(context), key, value);
    }

    public static void setScheduleAttribute(Context context, String key, String value) {
        setAttribute(spanFromContext(context), key, value);
    }

    public static void setScheduleAttribute(Context context, String key, boolean value) {
        setAttribute(spanFromContext(context), key, value);
    }

    public static void setScheduleDuration(Context context, String key,
                                            long startNanos, long endNanos) {
        if (startNanos > 0 && endNanos >= startNanos) {
            setScheduleAttribute(context, key,
                    TimeUnit.NANOSECONDS.toMillis(endNanos - startNanos));
        }
    }

    public static void finish(Span span) {
        finish(span, null);
    }

    public static void finish(Span span, Throwable error) {
        if (!enabled || span == null) {
            return;
        }
        BusinessError expectedError = null;
        try {
            // Inside the guard: the marker map is a synchronized WeakHashMap keyed
            // by Span, so it calls hashCode()/equals() on an implementation this
            // class does not control.
            expectedError = error == null ? BUSINESS_ERRORS.remove(span) : null;
            if (error == null) {
                if (expectedError == null) {
                    span.setStatus(StatusCode.OK);
                } else {
                    setBusinessError(span, expectedError.code(), expectedError.type());
                }
            } else {
                BUSINESS_ERRORS.remove(span);
                span.recordException(error);
                span.setStatus(StatusCode.ERROR, error.getClass().getSimpleName());
            }
        } catch (Throwable ignored) {
            // Exporter/provider failures are explicitly fail-open.
        } finally {
            try {
                span.end();
            } catch (Throwable ignored) {
                // Exporter/provider failures are explicitly fail-open.
            }
        }
    }

    /**
     * Ends a span this class owns using the call's real gRPC terminal status.
     *
     * <p>Deliberately not routed through {@link #finish(Span, Throwable)}: that
     * method sets OK when it finds no marker, and the SDK treats OK as final, so
     * an ERROR recorded here would be overwritten. A non-OK close is an error in
     * its own right even without a Throwable; a business-error marker still wins
     * because it names the cause more precisely than the transport code.
     *
     * <p>Takes primitives rather than io.grpc.Status so this module stays free of
     * a gRPC dependency.
     */
    public static void finishWithGrpcStatus(Span span, String canonicalCode, int numericCode, boolean ok) {
        if (!enabled || span == null) {
            return;
        }
        setAttribute(span, RPC_RESPONSE_STATUS_CODE, canonicalCode);
        setAttribute(span, GRPC_STATUS_CODE, numericCode);
        setAttribute(span, RTP_LLM_GRPC_STATUS_CODE, numericCode);
        if (ok) {
            finish(span, null);
            return;
        }
        try {
            BusinessError marker = BUSINESS_ERRORS.remove(span);
            if (marker != null) {
                setBusinessError(span, marker.code(), marker.type());
            } else {
                setAttribute(span, ERROR_TYPE, canonicalCode);
                span.setStatus(StatusCode.ERROR, canonicalCode);
            }
        } catch (Throwable ignored) {
            // Exporter/provider failures are explicitly fail-open.
        } finally {
            try {
                span.end();
            } catch (Throwable ignored) {
                // Exporter/provider failures are explicitly fail-open.
            }
        }
    }

    /** Marks an already-created span from an asynchronous scheduling callback. */
    public static void markBusinessError(Context context, long code, String type) {
        if (!enabled) {
            return;
        }
        // Whole body guarded: this runs on scheduling callbacks, and a throwing
        // Span or marker-map implementation must not reach the schedule response,
        // the gRPC status, or the routing result.
        try {
            Span span = spanFromContext(context);
            if (span.getSpanContext().isValid()) {
                // Apply the status now rather than only leaving a marker: for an
                // upstream-owned SERVER span (for example, auto-instrumentation or
                // another interceptor), GrpcTraceInterceptor keeps ownsSpan=false and
                // never calls finish() on it, so a marker alone would never be
                // consumed and its owner would end the span with no error recorded.
                // Keep the marker as well -- finish() on a span this class owns
                // consumes it and re-applies the error, which is what stops its
                // default setStatus(OK) from overwriting this ERROR (the SDK only
                // treats OK as final, so ERROR->OK does overwrite).
                setBusinessError(span, code, type);
                BUSINESS_ERRORS.put(span, new BusinessError(code, type));
            }
        } catch (Throwable ignored) {
            // Trace must never affect routing or dispatch.
        }
    }

    private static Span start(String name, Context parent, SpanKind kind) {
        if (!enabled) {
            return null;
        }
        try {
            Tracer tracer = GlobalOpenTelemetry.getTracer(INSTRUMENTATION_NAME);
            SpanBuilder builder = tracer.spanBuilder(name).setSpanKind(kind);
            if (parent == null) {
                builder.setParent(Context.current());
            } else {
                builder.setParent(parent);
            }
            return builder.startSpan();
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static Span spanFromContext(Context context) {
        try {
            return context == null ? Span.getInvalid() : Span.fromContext(context);
        } catch (Throwable ignored) {
            return Span.getInvalid();
        }
    }

    private static void setBusinessError(Span span, long code, String type) {
        String description = type == null ? "FLEXLB_BUSINESS_REJECTED" : type;
        setAttribute(span, ERROR_TYPE, description);
        setAttribute(span, SCHEDULE_CODE, code);
        span.setStatus(StatusCode.ERROR, description);
    }

    private record BusinessError(long code, String type) {
    }
}
