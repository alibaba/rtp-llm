package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract test for the per-rid prefill terminal trace line
 * ({@code mock_prefill_done rid=... ts_epoch_ms=...}) emitted by the batch
 * completion callback in {@code FastRpcService.runPrefillBatch}.
 *
 * <p>The line is the engine-side half of the BIRTH-axis engine-exec
 * percentiles: aggregate_canvas_run.py joins {@code exec_ms} against the load
 * client's {@code send_start_epoch_ms} bucket by {@code request_id} (the
 * numeric GenerateInputPB.requestId, NOT the trace-source rid string), so
 * prefill exec lands on the same axis as e2e/full_e2e. A format drift here
 * silently breaks that join, so the exact kv layout is asserted — if this
 * test fails after an intentional format change, update the parser in
 * aggregate_canvas_run.py in the same commit.
 *
 * <p>Prefill executes whole BATCHES, so the contract has two batch-shaped
 * properties beyond the kv layout: every member of one batch logs the SAME
 * exec_ms (the batch execution duration) and the SAME ts_epoch_ms (one
 * wall-clock stamp taken at batch completion). Both are asserted here via a
 * single 3-request enqueueBatch.
 *
 * <p>Observation channel: the test injects its own response queues into the
 * service's {@code responseQueues} map (enqueueBatch's computeIfAbsent keeps
 * pre-registered entries), and the trace line is printed BEFORE the terminal
 * frame is offered on the same completion-callback iteration (no decode role
 * addr configured → startDecode returns false → finished=true frame offered
 * from the prefill side) — so each frame's arrival proves that member's line
 * was already written.
 *
 * <p>The {@code cancelled=true} branch (a cancel marker landing before the
 * batch completes) is a race window that cannot be reproduced
 * deterministically; only the normal {@code cancelled=false} output is
 * asserted here.
 */
class PrefillDoneLogLineTest {

    /** Exact wire contract — keep in sync with aggregate_canvas_run.py. */
    private static final Pattern LINE_RE = Pattern.compile(
            "^mock_prefill_done rid=(-?\\d+) ts_epoch_ms=(\\d+) exec_ms=(\\d+)"
                    + " input_len=(\\d+) cancelled=(true|false)$");

    private static final int BASE_PORT = 63600;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(2, runnable -> {
            Thread thread = new Thread(runnable, "prefill-done-log-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        nextPortOffset = 0;
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    @Test
    @Timeout(30)
    void prefillBatchCompletionEmitsPerRidTerminalLine() throws Exception {
        // Flat 20 ms prefill formula at sleep_scale 0.1 → one batch books
        // ~2 ms regardless of batch size, so the 3-member batch completes in
        // milliseconds — comfortably observable, fast to wait.
        MockPerformanceModel model = MockEngineTestSupport.performanceModel(
                tempDir, "20", 0.1, 1000.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model);

        long[] rids = {5151L, 5152L, 5153L};
        int[] inputLens = {10, 20, 30};
        // Hold DIRECT queue references: the batch completion callback removes
        // the entry from responseQueues after offering the terminal frame (per-
        // request state cleanup), so re-fetching from the map would NPE once
        // the callback has already run (batch books ~2 ms).
        @SuppressWarnings("unchecked")
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[] queues =
                (LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[])
                        new LinkedBlockingQueue[rids.length];
        Map<Long, LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>> responseQueues =
                responseQueuesOf(prefill);
        for (int i = 0; i < rids.length; i++) {
            queues[i] = new LinkedBlockingQueue<>();
            responseQueues.put(rids[i], queues[i]);
        }

        long beforeMs = System.currentTimeMillis();
        PrintStream originalOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream capture = new PrintStream(captured, true, StandardCharsets.UTF_8);
        System.setOut(capture);
        try {
            EngineRpcService.EnqueueBatchRequestPB batch = MockEngineTestSupport.batch(
                    777L,
                    MockEngineTestSupport.slot(
                            0,
                            MockEngineTestSupport.input(rids[0], inputLens[0]),
                            MockEngineTestSupport.input(rids[1], inputLens[1]),
                            MockEngineTestSupport.input(rids[2], inputLens[2])));
            EngineRpcService.EnqueueBatchResponsePB response =
                    MockEngineTestSupport.enqueue(prefill, batch);
            assertEquals(
                    rids.length,
                    response.getSuccessesCount(),
                    "all batch members must be admitted, got: " + response);
            // One terminal frame per member proves its trace line was written
            // (the printf precedes the frame offer in the same callback
            // iteration).
            for (LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue : queues) {
                EngineRpcService.GenerateOutputsPB frame =
                        queue.poll(10, TimeUnit.SECONDS);
                assertNotNull(
                        frame,
                        "terminal frame must arrive within 10s");
                assertTrue(
                        frame.getFlattenOutput().getFinished(0),
                        "frame must be the prefill-side terminal frame");
            }
        } finally {
            System.setOut(originalOut);
            capture.flush();
        }
        long afterMs = System.currentTimeMillis();

        String capturedLog = captured.toString(StandardCharsets.UTF_8);
        List<String> traceLines = capturedLog.lines()
                .filter(line -> line.startsWith("mock_prefill_done"))
                .toList();
        assertEquals(
                rids.length,
                traceLines.size(),
                "exactly one mock_prefill_done line per batch member, got: " + capturedLog);

        Map<Long, Matcher> byRid = new LinkedHashMap<>();
        for (String line : traceLines) {
            Matcher matcher = LINE_RE.matcher(line);
            assertTrue(
                    matcher.matches(),
                    "line must match the exact kv contract (update aggregate_canvas_run.py"
                            + " parser in the same commit on intentional change): " + line);
            byRid.put(Long.parseLong(matcher.group(1)), matcher);
        }
        assertEquals(rids.length, byRid.size(), "each line must carry a distinct rid");

        long execMsFirst = -1L;
        long tsFirst = -1L;
        for (int i = 0; i < rids.length; i++) {
            Matcher matcher = byRid.get(rids[i]);
            assertNotNull(matcher, "missing trace line for rid " + rids[i]);

            long ts = Long.parseLong(matcher.group(2));
            assertTrue(
                    ts >= beforeMs - 1 && ts <= afterMs + 1,
                    "ts_epoch_ms must be an epoch-ms wall-clock timestamp inside the test"
                            + " window [" + beforeMs + ", " + afterMs + "], got " + ts);

            long execMs = Long.parseLong(matcher.group(3));
            // exec_ms caliber: the BATCH execution duration (prefill runs
            // whole batches), i.e. formula "20" x sleep_scale 0.1 = 2 ms with
            // the scaledMs floor of 1. Assert the floor plus batch-uniformity
            // below; the exact model arithmetic is not the contract here.
            assertTrue(
                    execMs >= 1,
                    "exec_ms must be the batch execution duration (>= scaled floor), got "
                            + execMs);
            if (i == 0) {
                execMsFirst = execMs;
                tsFirst = ts;
            } else {
                assertEquals(
                        execMsFirst,
                        execMs,
                        "members of one batch must log the SAME exec_ms (batch duration)");
                assertEquals(
                        tsFirst,
                        ts,
                        "members of one batch must log the SAME ts_epoch_ms (one"
                                + " completion stamp)");
            }

            assertEquals(
                    inputLens[i],
                    Integer.parseInt(matcher.group(4)),
                    "input_len must echo the request's input token count");
            assertEquals("false", matcher.group(5), "normal completion must log cancelled=false");
        }
    }

    // ──────────── helpers ────────────

    private JavaMockEngineCluster.FastRpcService newPrefillService(MockPerformanceModel model) {
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill-done-log-" + port, "127.0.0.1", "prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                port, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(), 10_000_000L, 8);
        services.put(port, service);
        return service;
    }

    @SuppressWarnings("unchecked")
    private static Map<Long, LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>> responseQueuesOf(
            JavaMockEngineCluster.FastRpcService service) throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class.getDeclaredField("responseQueues");
        field.setAccessible(true);
        return (Map<Long, LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>>)
                field.get(service);
    }
}
