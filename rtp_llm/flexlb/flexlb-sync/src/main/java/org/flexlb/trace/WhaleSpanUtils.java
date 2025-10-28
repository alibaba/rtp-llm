package org.flexlb.trace;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.dao.RequestContext;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

import static org.flexlb.constant.HttpHeaderNames.TRACE_PARENT;
import static org.flexlb.constant.HttpHeaderNames.TRACE_STATE;

/**
 * @author laichuan
 * @date 2025/8/21
 **/
@Slf4j
@Component
public class WhaleSpanUtils {

    private static final String TRACE_SPAN_ENABLED_ENV = "WHALE_TRACE_SPAN_ENABLED";
    private static final boolean TRACE_SPAN_ENABLED;

    static {
        String envValue = System.getenv(TRACE_SPAN_ENABLED_ENV);
        if (envValue == null) {
            TRACE_SPAN_ENABLED = true;
        } else {
            TRACE_SPAN_ENABLED = "1".equals(envValue);
        }
        log.info("TraceSpan enabled: {}", TRACE_SPAN_ENABLED);
    }

    public void buildTraceSpan(RequestContext ctx) {
        WhaleSpan whaleSpan = createSpan(ctx);
        ctx.setSpan(whaleSpan);
    }

    public static WhaleSpan createSpan(RequestContext ctx) {

        if (!TRACE_SPAN_ENABLED) {
            return new NoopSpanImpl();
        }

        boolean isNotRecord = checkNotRecordSpan(ctx);
        if (isNotRecord) {
            return new NoopSpanImpl();
        }

        Map<String, String> spanCarrier = new HashMap<>();

        //请求来自接入层，只有纯OTLP一种传输方式
        if (StringUtils.isNotBlank(ctx.getOtlpTraceParent())) {
            spanCarrier.put(TRACE_PARENT, ctx.getOtlpTraceParent());
        }
        if (StringUtils.isNotBlank(ctx.getOtlpTraceState())) {
            spanCarrier.put(TRACE_STATE, ctx.getOtlpTraceState());
        }

        // OpenTelemetry SpanName语义约定: https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/
        WhaleSpan whaleSpan = new WhaleSpanImpl("master");
        whaleSpan.startSpan(spanCarrier);
        return whaleSpan;
    }

    private static boolean checkNotRecordSpan(RequestContext ctx) {
        return ctx.isPrivateRequest();
    }
}
