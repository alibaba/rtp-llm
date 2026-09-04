package org.flexlb.telemetry;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanBuilder;
import io.opentelemetry.api.trace.SpanContext;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.propagation.TextMapGetter;
import io.opentelemetry.context.propagation.TextMapPropagator;
import io.opentelemetry.context.propagation.TextMapSetter;
import io.opentelemetry.api.trace.propagation.W3CTraceContextPropagator;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.WeakHashMap;

/**
 * Small fail-open facade for the FlexLB manual tracing points.
 *
 * <p>The production trace agent owns the SDK/exporter configuration. When no
 * agent/provider is installed, the global OpenTelemetry implementation is a
 * no-op and these methods leave scheduling behavior unchanged.</p>
 */
public final class FlexlbTrace {

    public static final String INSTRUMENTATION_NAME = "org.flexlb";
    public static final String REQUEST_ID = "rtp_llm.request_id";
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

    private static final TextMapSetter<Map<String, String>> MAP_SETTER =
            (carrier, key, value) -> carrier.put(key, value);
    private static final TextMapPropagator W3C_PROPAGATOR =
            W3CTraceContextPropagator.getInstance();
    // Weak keys cover agent-owned SERVER spans too: the manual interceptor
    // must not retain a marker forever when an external agent owns ending.
    private static final Map<Span, BusinessError> BUSINESS_ERRORS =
            Collections.synchronizedMap(new WeakHashMap<>());

    private FlexlbTrace() {
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
            return span == null ? base : span.storeInContext(base);
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
        setAttribute(span, REQUEST_ID, requestId);
    }

    public static void setAttribute(Span span, String key, long value) {
        try {
            if (span != null) {
                span.setAttribute(key, value);
            }
        } catch (Throwable ignored) {
            // Trace must never affect routing or dispatch.
        }
    }

    public static void setAttribute(Span span, String key, String value) {
        try {
            if (span != null && value != null) {
                span.setAttribute(key, value);
            }
        } catch (Throwable ignored) {
            // Trace must never affect routing or dispatch.
        }
    }

    public static void setAttribute(Span span, String key, boolean value) {
        try {
            if (span != null) {
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
        if (span == null) {
            return;
        }
        BusinessError expectedError = error == null ? BUSINESS_ERRORS.remove(span) : null;
        try {
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

    /** Marks an already-created span from an asynchronous scheduling callback. */
    public static void markBusinessError(Context context, long code, String type) {
        Span span = spanFromContext(context);
        if (span.getSpanContext().isValid()) {
            BUSINESS_ERRORS.put(span, new BusinessError(code, type));
        }
    }

    private static Span start(String name, Context parent, SpanKind kind) {
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
            return Span.getInvalid();
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
        setAttribute(span, "error.type", description);
        setAttribute(span, SCHEDULE_CODE, code);
        span.setStatus(StatusCode.ERROR, description);
    }

    private record BusinessError(long code, String type) {
    }
}
