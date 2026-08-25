package org.flexlb.httpserver;

import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Periodically dumps the cumulative master arrival/completion counters of
 * {@link ServerScheduleLatencyRecorder} to a file, one line per tick:
 *
 * <pre>ts_epoch_ms=&lt;epoch-ms&gt; arrival_count=&lt;n&gt; completion_count=&lt;n&gt;</pre>
 *
 * <p>The line format is field-compatible with the master_counters_timeseries.txt
 * files previously written by the online-eval Python poller, so downstream
 * tooling keeps working unchanged. The file is opened in append mode and
 * flushed on every tick; rotation/cleanup is owned by the consumer (each eval
 * run uses a fresh run directory).
 *
 * <p>Configuration (Spring relaxed binding from environment variables):
 * <ul>
 *   <li>{@code flexlb.counter-dump.path} / {@code FLEXLB_COUNTER_DUMP_PATH}:
 *       output file path. Empty (default) disables the dumper entirely — the
 *       file is never rotated here, so production masters must opt in and the
 *       caller owns cleanup.</li>
 *   <li>{@code flexlb.counter-dump.interval-ms} /
 *       {@code FLEXLB_COUNTER_DUMP_INTERVAL_MS}: dump interval in milliseconds,
 *       default 1000. Values &lt;= 0 disable the dumper.</li>
 * </ul>
 */
@Slf4j
@Component
public class ServerScheduleLatencyCounterDumper {

    private final ServerScheduleLatencyRecorder recorder;
    private final ScheduledExecutorService executor;
    private final BufferedWriter writer;

    public ServerScheduleLatencyCounterDumper(
            ServerScheduleLatencyRecorder recorder,
            @Value("${flexlb.counter-dump.path:}") String dumpPath,
            @Value("${flexlb.counter-dump.interval-ms:1000}") long intervalMs) {
        this.recorder = recorder;
        if (isBlank(dumpPath) || intervalMs <= 0) {
            this.executor = null;
            this.writer = null;
            return;
        }
        this.writer = openWriter(dumpPath);
        this.executor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "flexlb-counter-dumper");
            thread.setDaemon(true);
            return thread;
        });
        this.executor.scheduleAtFixedRate(this::dumpCounters, 0, intervalMs, TimeUnit.MILLISECONDS);
        log.info("flexlb counter dump enabled: path={} interval_ms={}", dumpPath, intervalMs);
    }

    /**
     * Appends one counter line for the current recorder state. Package-visible
     * for deterministic unit tests; normally invoked by the scheduler only.
     */
    void dumpCounters() {
        if (writer == null) {
            return;
        }
        try {
            Map<String, Object> snapshot = recorder.snapshot();
            writer.write("ts_epoch_ms=" + System.currentTimeMillis()
                    + " arrival_count=" + snapshot.get("arrival_count")
                    + " completion_count=" + snapshot.get("completion_count") + "\n");
            writer.flush();
        } catch (Exception e) {
            // Never let an exception kill the scheduled task; retry next tick.
            log.warn("flexlb counter dump write failed, will retry next tick", e);
        }
    }

    @PreDestroy
    void stop() {
        if (executor != null) {
            executor.shutdownNow();
        }
        if (writer != null) {
            // Final dump so at most one interval of trailing samples is lost
            // when the master receives a graceful shutdown signal.
            dumpCounters();
            try {
                writer.close();
            } catch (IOException e) {
                log.warn("flexlb counter dump close failed", e);
            }
        }
    }

    private static BufferedWriter openWriter(String dumpPath) {
        try {
            Path path = Path.of(dumpPath);
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            return Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            throw new IllegalStateException("cannot open flexlb counter dump file: " + dumpPath, e);
        }
    }

    private static boolean isBlank(String value) {
        return value == null || value.trim().isEmpty();
    }
}
