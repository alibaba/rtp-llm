package org.flexlb.httpserver;

import org.flexlb.dao.BalanceContext;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ServerScheduleLatencyCounterDumperTest {

    private static final Pattern LINE_PATTERN = Pattern.compile(
            "^ts_epoch_ms=\\d+ arrival_count=\\d+ completion_count=\\d+$");

    @TempDir
    Path tempDir;

    @Test
    void writesFieldCompatibleCounterLinesAndFinalDumpOnStop() throws Exception {
        Path file = tempDir.resolve("counters.log");
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        recordOneArrivalAndCompletion(recorder);
        recordOneArrival(recorder);

        // Long interval: only the t=0 initial dump runs in the background;
        // manual dumpCounters() calls below control the rest deterministically.
        ServerScheduleLatencyCounterDumper dumper =
                new ServerScheduleLatencyCounterDumper(recorder, file.toString(), 60_000);
        dumper.dumpCounters();

        List<String> lines = Files.readAllLines(file);
        assertTrue(lines.size() >= 2);
        for (String line : lines) {
            assertTrue(LINE_PATTERN.matcher(line).matches(), "bad line: " + line);
        }
        assertTrue(containsCounters(lines, 2, 1), "missing arrival=2 completion=1 line in " + lines);

        // @PreDestroy must flush one final line with the latest counters.
        recordOneArrival(recorder);
        dumper.stop();
        lines = Files.readAllLines(file);
        String last = lines.get(lines.size() - 1);
        assertTrue(LINE_PATTERN.matcher(last).matches(), "bad final line: " + last);
        assertTrue(containsCounters(List.of(last), 3, 1), "final line misses latest counters: " + last);
    }

    @Test
    void appendsAcrossDumperInstancesInsteadOfTruncating() throws Exception {
        Path file = tempDir.resolve("counters.log");
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();

        ServerScheduleLatencyCounterDumper first =
                new ServerScheduleLatencyCounterDumper(recorder, file.toString(), 60_000);
        first.dumpCounters();
        first.stop();
        int afterFirst = Files.readAllLines(file).size();
        assertTrue(afterFirst >= 1);

        // A new master process (new dumper instance) must keep appending, matching
        // the Python poller's append-mode semantics on reused run directories.
        ServerScheduleLatencyCounterDumper second =
                new ServerScheduleLatencyCounterDumper(recorder, file.toString(), 60_000);
        second.dumpCounters();
        second.stop();
        assertTrue(Files.readAllLines(file).size() > afterFirst);
    }

    @Test
    void dumpsPeriodicallyAtConfiguredInterval() throws Exception {
        Path file = tempDir.resolve("counters.log");
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();

        ServerScheduleLatencyCounterDumper dumper =
                new ServerScheduleLatencyCounterDumper(recorder, file.toString(), 50);
        try {
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
            while (Files.readAllLines(file).size() < 2 && System.nanoTime() < deadline) {
                TimeUnit.MILLISECONDS.sleep(20);
            }
            List<String> lines = Files.readAllLines(file);
            assertTrue(lines.size() >= 2, "scheduler did not dump at interval: " + lines);
            for (String line : lines) {
                assertTrue(LINE_PATTERN.matcher(line).matches(), "bad line: " + line);
            }
        } finally {
            dumper.stop();
        }
    }

    @Test
    void blankPathDisablesDumper() throws Exception {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        ServerScheduleLatencyCounterDumper dumper =
                new ServerScheduleLatencyCounterDumper(recorder, "", 1);
        TimeUnit.MILLISECONDS.sleep(200);
        assertFalse(Files.exists(tempDir.resolve("counters.log")));
        // Disabled instance must be a safe no-op, not an NPE.
        dumper.dumpCounters();
        dumper.stop();
    }

    @Test
    void nonPositiveIntervalDisablesDumper() throws Exception {
        Path file = tempDir.resolve("counters.log");
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();

        ServerScheduleLatencyCounterDumper zeroInterval =
                new ServerScheduleLatencyCounterDumper(recorder, file.toString(), 0);
        ServerScheduleLatencyCounterDumper negativeInterval =
                new ServerScheduleLatencyCounterDumper(recorder, file.toString(), -1);
        TimeUnit.MILLISECONDS.sleep(200);
        zeroInterval.dumpCounters();
        negativeInterval.dumpCounters();
        zeroInterval.stop();
        negativeInterval.stop();
        assertFalse(Files.exists(file));
    }

    private static void recordOneArrival(ServerScheduleLatencyRecorder recorder) {
        recorder.recordArrival(System.nanoTime());
    }

    private static void recordOneArrivalAndCompletion(ServerScheduleLatencyRecorder recorder) {
        long end = System.nanoTime();
        BalanceContext context = new BalanceContext();
        context.setGrpcEntryNanos(end - TimeUnit.MILLISECONDS.toNanos(20));
        context.setServiceStartNanos(end - TimeUnit.MILLISECONDS.toNanos(18));
        context.setRouteSubmittedNanos(end - TimeUnit.MILLISECONDS.toNanos(15));
        context.setBatchDispatchedNanos(end - TimeUnit.MILLISECONDS.toNanos(10));
        context.setAckAtNanos(end - TimeUnit.MILLISECONDS.toNanos(2));
        recorder.recordArrival(end);
        recorder.recordCompletion(context, end);
    }

    private static boolean containsCounters(List<String> lines, long arrival, long completion) {
        return lines.stream().anyMatch(line -> line.contains(" arrival_count=" + arrival + " ")
                && line.endsWith(" completion_count=" + completion));
    }
}
