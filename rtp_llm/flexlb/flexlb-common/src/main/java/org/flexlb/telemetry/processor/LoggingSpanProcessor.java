package org.flexlb.telemetry.processor;

import io.opentelemetry.context.Context;
import io.opentelemetry.sdk.trace.ReadWriteSpan;
import io.opentelemetry.sdk.trace.ReadableSpan;
import io.opentelemetry.sdk.trace.SpanProcessor;
import io.opentelemetry.sdk.trace.data.SpanData;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class LoggingSpanProcessor implements SpanProcessor {

    public static LoggingSpanProcessor create() {
        return new LoggingSpanProcessor();
    }

    @Override
    public void onStart(Context context, ReadWriteSpan readWriteSpan) {
        SpanData spanData = readWriteSpan.toSpanData();
        log.info("[span on start] name:{} traceId:{} spanId:{} parentId:{}", spanData.getName(), spanData.getTraceId(), spanData.getSpanId(), spanData.getParentSpanId());
    }

    @Override
    public boolean isStartRequired() {
        return true;
    }

    @Override
    public void onEnd(ReadableSpan readableSpan) {
        SpanData spanData = readableSpan.toSpanData();
        log.info("[span on end] name:{} traceId:{} spanId:{} parentId:{}", spanData.getName(), spanData.getTraceId(), spanData.getSpanId(), spanData.getParentSpanId());
    }

    @Override
    public boolean isEndRequired() {
        return true;
    }
}
