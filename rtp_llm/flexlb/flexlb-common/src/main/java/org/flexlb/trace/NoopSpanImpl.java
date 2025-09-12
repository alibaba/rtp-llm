package org.flexlb.trace;

import io.opentelemetry.context.Scope;

import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * @author laichuan
 * @date 2025/8/19
 **/
public class NoopSpanImpl implements WhaleSpan {

    @Override
    public Scope makeCurrent() {
        return null;
    }

    @Override
    public void endSpan() {

    }

    @Override
    public void endSpan(long timestamp, TimeUnit unit) {

    }

    @Override
    public void setAttribute(String key, String value) {

    }

    @Override
    public void addEvent(String name) {

    }

    @Override
    public void addEvent(String name, Map<String, Object> attributes) {

    }

    @Override
    public void recordException(Throwable exception) {

    }

    @Override
    public void setStatus(String statusCode) {

    }

    @Override
    public void addAttributeOfEvent(String key) {

    }

    @Override
    public void addAttributeOfEvent(String key, Object value) {

    }

    @Override
    public void pushOutUnSubmittedAttributes() {

    }

    @Override
    public String getTraceId() {
        return "";
    }

    @Override
    public void startSpan(Map<String, String> carrier) {

    }
}
