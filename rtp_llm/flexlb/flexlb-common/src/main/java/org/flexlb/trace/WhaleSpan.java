package org.flexlb.trace;

import io.opentelemetry.context.Scope;

import java.util.Map;
import java.util.concurrent.TimeUnit;

public interface WhaleSpan {
    void startSpan(Map<String, String> carrier);

    Scope makeCurrent();

    void endSpan();

    void endSpan(long timestamp, TimeUnit unit);

    void setAttribute(String key, String value);

    void addEvent(String name);

    void addEvent(String name, Map<String, Object> attributes);

    void recordException(Throwable exception);

    void setStatus(String statusCode);

    void addAttributeOfEvent(String key);

    void addAttributeOfEvent(String key, Object value);

    void pushOutUnSubmittedAttributes();

    String getTraceId();
}
