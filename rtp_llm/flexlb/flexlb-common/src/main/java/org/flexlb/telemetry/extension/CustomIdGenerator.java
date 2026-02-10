package org.flexlb.telemetry.extension;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.TraceId;
import io.opentelemetry.sdk.trace.IdGenerator;

import java.net.InetAddress;

/**
 * Custom trace ID generator
 */
public class CustomIdGenerator implements IdGenerator {

    public static final String[] APPEND_ZEROS = makeConst();

    public static final String[] PADDING_ZEROS = makePaddingZeros();

    public static final IdGenerator randomGenerator = IdGenerator.random();

    private static String[] makeConst() {
        String[] tmp = new String[33];

        StringBuilder buf = new StringBuilder(tmp.length);
        for (int i = 0; i < tmp.length; i++) {
            tmp[i] = buf.toString();
            buf.append("0");
        }
        return tmp;
    }

    private static String[] makePaddingZeros() {
        String[] result = new String[33];
        for (int i = 0; i < 33; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < i; j++) {
                sb.append('0');
            }
            result[i] = sb.toString();
        }
        return result;
    }

    @Override
    public String generateSpanId() {
        return randomGenerator.generateSpanId();
    }

    @Override
    public String generateTraceId() {
        String traceId = Span.current().getSpanContext().getTraceId();
        if (traceId.length() == TraceId.getLength()) {
            return traceId;
        }

        return traceId + APPEND_ZEROS[TraceId.getLength() - traceId.length()];
    }

    public String getIP() {
        try {
            return InetAddress.getLocalHost().getHostAddress();
        } catch (Exception e) {
            return "unknown";
        }
    }
}