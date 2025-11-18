package org.flexlb.telemetry.exporter;

import io.opentelemetry.sdk.common.CompletableResultCode;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SpanExporter;
import lombok.extern.java.Log;
import org.springframework.lang.Nullable;

import java.util.Collection;
import java.util.logging.Handler;

@Log
public class LoggingSpanExporter implements SpanExporter {

    /**
     * Returns a new {@link LoggingSpanExporter}.
     */
    public static LoggingSpanExporter create() {
        return new LoggingSpanExporter();
    }

    @Override
    public CompletableResultCode export(@Nullable Collection<SpanData> spans) {
        return CompletableResultCode.ofSuccess();
    }

    /**
     * Flushes the data.
     *
     * @return the result of the operation
     */
    @Override
    public CompletableResultCode flush() {
        CompletableResultCode resultCode = new CompletableResultCode();
        for (Handler handler : log.getHandlers()) {
            try {
                handler.flush();
            } catch (Throwable t) {
                resultCode.fail();
            }
        }
        return resultCode.succeed();
    }

    @Override
    public CompletableResultCode shutdown() {
        return flush();
    }
}
