package org.flexlb.trace;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.common.Attributes;
import io.opentelemetry.api.common.AttributesBuilder;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.SpanBuilder;
import io.opentelemetry.api.trace.SpanKind;
import io.opentelemetry.api.trace.StatusCode;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Context;
import io.opentelemetry.context.Scope;
import io.opentelemetry.context.propagation.TextMapGetter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.collections4.MapUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.constant.CommonConstants;

import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.TimeUnit;

/**
 * @author laichuan
 **/
@Slf4j
public class WhaleSpanImpl implements WhaleSpan {
    private static final Long MAX_ATTRIBUTE_NUM_OF_EVENT = 128L;
    private static final Tracer TPP_TRACER = GlobalOpenTelemetry.get().getTracer("tpp");
    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS");
    private static final ZoneId ZONE_ID = ZoneId.systemDefault();

    private Span span;
    private final SpanBuilder spanBuilder;
    private final Map<String, Object> unSubmittedAttributes = new ConcurrentSkipListMap<>();

    public WhaleSpanImpl(String name) {
        spanBuilder = TPP_TRACER.spanBuilder(name);
    }

    public void startSpan(Map<String, String> carrier) {

        TextMapGetter<Map<String, String>> getter =
                new TextMapGetter<>() {
                    @Override
                    public String get(Map<String, String> carrier, String key) {
                        if (carrier == null) {
                            return null;
                        }
                        if (carrier.containsKey(key)) {
                            return carrier.get(key);
                        }
                        return null;
                    }

                    @Override
                    public Iterable<String> keys(Map<String, String> carrier) {
                        return carrier.keySet();
                    }
                };

        Context extractedContext = GlobalOpenTelemetry.getPropagators().getTextMapPropagator().extract(Context.root(), carrier, getter);
        span = spanBuilder.setParent(extractedContext).setSpanKind(SpanKind.SERVER).startSpan();
        span.addEvent(CommonConstants.START);
    }

    @Override
    public Scope makeCurrent() {
        return span.makeCurrent();
    }

    public void endSpan() {
        pushOutUnSubmittedAttributes();
        span.end();
    }

    public void endSpan(long timestamp, TimeUnit unit) {
        pushOutUnSubmittedAttributes();
        span.end(timestamp, unit);
    }

    public void setAttribute(String key, String value) {
        span.setAttribute(key, value);
    }

    public void addEvent(String name) {
        span.addEvent(withThreadName(name));
    }

    public void addEvent(String name, Map<String, Object> attributes) {
        span.addEvent(withThreadName(name), convertAttributes(attributes));
    }

    public void recordException(Throwable exception) {
        pushOutUnSubmittedAttributes();
        span.setStatus(StatusCode.ERROR);
        span.recordException(exception);
    }

    public void setStatus(String statusCode) {
        if (StringUtils.isEmpty(statusCode)) {
            return;
        }
        try {
            span.setStatus(StatusCode.valueOf(statusCode));
        } catch (Throwable e) {
//            TraceLogger.error("Failed to set status code: {}", statusCode, e);
        }
    }

    public void addAttributeOfEvent(String key) {
        addAttributeOfEvent(key, "");
    }

    public void addAttributeOfEvent(String key, Object value) {
        if (StringUtils.isEmpty(key) || Objects.isNull(value)) {
            return;
        }
        key = FORMATTER.format(LocalDateTime.ofInstant(Instant.ofEpochMilli(System.currentTimeMillis()), ZONE_ID)) + " " + key;

        unSubmittedAttributes.put(key, value);
        if (unSubmittedAttributes.size() >= MAX_ATTRIBUTE_NUM_OF_EVENT) {
            span.addEvent("EventAggregation", convertAttributes(unSubmittedAttributes));
            unSubmittedAttributes.clear();
        }
    }

    private Attributes convertAttributes(Map<String, Object> attributes) {
        AttributesBuilder builder = Attributes.builder();
        for (Map.Entry<String, Object> entry : Optional.ofNullable(attributes).orElse(new HashMap<>()).entrySet()) {
            if (StringUtils.isEmpty(entry.getKey()) || Objects.isNull(entry.getValue())) {
                continue;
            }
            builder.put(entry.getKey(), entry.getValue().toString());
        }
        return builder.build();
    }

    public void pushOutUnSubmittedAttributes() {
        if (MapUtils.isNotEmpty(unSubmittedAttributes)) {
            span.addEvent("EventAggregation", convertAttributes(unSubmittedAttributes));
            unSubmittedAttributes.clear();
        }
    }

    public String getTraceId() {
        return span.getSpanContext().getTraceId();
    }

    private String withThreadName(String s) {
        return Thread.currentThread().getName() + " " + s;
    }
}
