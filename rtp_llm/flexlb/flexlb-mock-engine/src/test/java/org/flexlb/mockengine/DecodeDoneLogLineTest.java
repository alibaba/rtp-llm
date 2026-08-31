package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.reflect.Method;
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
 * Contract test for the per-rid decode terminal trace line
 * ({@code mock_decode_done rid=... ts_epoch_ms=...}) emitted by
 * {@code FastRpcService.publishDecodeCompletion}.
 *
 * <p>The line is the engine-side half of the schedule-only full-e2e metric:
 * aggregate_canvas_run.py joins {@code ts_epoch_ms} against the load client's
 * {@code send_start_epoch_ms} by {@code request_id} (the numeric
 * GenerateInputPB.requestId, NOT the trace-source rid string). A format drift
 * here silently breaks that join, so the exact kv layout is asserted — if
 * this test fails after an intentional format change, update the parser in
 * aggregate_canvas_run.py in the same commit.
 *
 * <p>Observation channel: each request is admitted through the reflection
 * admission point with its OWN response queue (direct generate_stream path),
 * and the trace line is printed BEFORE the terminal frame is offered on the
 * same thread — so the frame's arrival proves the line was already written.
 *
 * <p>The {@code cancelled=true} branch (a cancel marker landing between the
 * terminal claim and the publish) is a narrow race window that cannot be
 * reproduced deterministically; only the normal {@code cancelled=false}
 * output is asserted here.
 */
class DecodeDoneLogLineTest {

    /** Exact wire contract — keep in sync with aggregate_canvas_run.py. */
    private static final Pattern LINE_RE = Pattern.compile(
            "^mock_decode_done rid=(-?\\d+) ts_epoch_ms=(\\d+) exec_ms=(\\d+)"
                    + " output_len=(\\d+) cancelled=(true|false)$");

    private static final int BASE_PORT = 63500;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(2, runnable -> {
            Thread thread = new Thread(runnable, "decode-done-log-scheduler");
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
    void decodeCompletionEmitsPerRidTerminalLine() throws Exception {
        // Flat 20 ms model steps at sleep_scale 0.1 (decodeModel default):
        // each step books/sleeps ~2 ms, so every stream finishes in tens of
        // ms — comfortably observable, fast to wait.
        MockPerformanceModel model = MockEngineTestSupport.decodeModel(tempDir, 20.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 8);
        long beforeMs = System.currentTimeMillis();

        PrintStream originalOut = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        PrintStream capture = new PrintStream(captured, true, StandardCharsets.UTF_8);
        System.setOut(capture);
        try {
            long[] rids = {4242L, 4243L, 4244L};
            int[] outputLens = {4, 6, 2};
            @SuppressWarnings("unchecked")
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[] queues =
                    (LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[])
                            new LinkedBlockingQueue[rids.length];
            for (int i = 0; i < rids.length; i++) {
                queues[i] = new LinkedBlockingQueue<>();
                assertTrue(invokeScheduleDecodeCompletion(
                        decode, shapeOf(model, rids[i], 10, outputLens[i]), -1, queues[i]),
                        "request " + rids[i] + " must be admitted");
            }
            for (LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue : queues) {
                EngineRpcService.GenerateOutputsPB frame =
                        queue.poll(10, TimeUnit.SECONDS);
                assertNotNull(frame, "terminal frame must arrive within 10s");
                assertTrue(frame.getFlattenOutput().getFinished(0),
                        "frame must be the terminal frame");
            }
        } finally {
            System.setOut(originalOut);
            capture.flush();
        }
        long afterMs = System.currentTimeMillis();

        String capturedLog = captured.toString(StandardCharsets.UTF_8);
        List<String> traceLines = capturedLog.lines()
                .filter(line -> line.startsWith("mock_decode_done"))
                .toList();
        assertEquals(3, traceLines.size(),
                "exactly one mock_decode_done line per completed stream, got: " + capturedLog);

        Map<Long, Matcher> byRid = new LinkedHashMap<>();
        for (String line : traceLines) {
            Matcher matcher = LINE_RE.matcher(line);
            assertTrue(matcher.matches(),
                    "line must match the exact kv contract (update aggregate_canvas_run.py"
                            + " parser in the same commit on intentional change): " + line);
            byRid.put(Long.parseLong(matcher.group(1)), matcher);
        }
        assertEquals(3, byRid.size(), "each line must carry a distinct rid");
        long[] expectedRids = {4242L, 4243L, 4244L};
        int[] expectedOutputLens = {4, 6, 2};
        for (int i = 0; i < expectedRids.length; i++) {
            Matcher matcher = byRid.get(expectedRids[i]);
            assertNotNull(matcher, "missing trace line for rid " + expectedRids[i]);

            long ts = Long.parseLong(matcher.group(2));
            assertTrue(ts >= beforeMs - 1 && ts <= afterMs + 1,
                    "ts_epoch_ms must be an epoch-ms wall-clock timestamp inside the test"
                            + " window [" + beforeMs + ", " + afterMs + "], got " + ts);

            long execMs = Long.parseLong(matcher.group(3));
            // exec_ms caliber: the summed BOOKED step durations — each step
            // prices at model step_ms × sleep_scale (decodeModel() runs
            // sleep_scale=0.1, so one 20ms step books 2ms), and steps fold
            // tokens_per_step tokens (MTP), e.g. output_len=4 → 2 steps × 2ms
            // = 4ms. Assert only the one-booked-step lower bound; the exact
            // sum depends on tokens_per_step and is not the contract here.
            assertTrue(execMs >= 2,
                    "exec_ms must be the summed booked step durations "
                            + "(>= one 20ms*0.1 step), got " + execMs);

            assertEquals(expectedOutputLens[i], Integer.parseInt(matcher.group(4)),
                    "output_len must echo the stream's output token budget");
            assertEquals("false", matcher.group(5),
                    "normal completion must log cancelled=false");
        }
    }

    // ──────────── helpers (mirror ContinuousBatchingDecodeTest) ────────────

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int decodeMaxConcurrency) {
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "decode-done-log-" + port, "127.0.0.1", "decode",
                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                port, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(), 10_000_000L, decodeMaxConcurrency);
        services.put(port, service);
        return service;
    }

    private static MockPerformanceModel.RequestShape shapeOf(
            MockPerformanceModel model, long requestId, int inputTokens, int outputTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(outputTokens)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return model.shape(input.build(), new MockLruBlockCache(100));
    }

    private static boolean invokeScheduleDecodeCompletion(
            JavaMockEngineCluster.FastRpcService service,
            MockPerformanceModel.RequestShape shape,
            long batchId,
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue)
            throws Exception {
        Method method = JavaMockEngineCluster.FastRpcService.class.getDeclaredMethod(
                "scheduleDecodeCompletion",
                MockPerformanceModel.RequestShape.class,
                long.class,
                LinkedBlockingQueue.class);
        method.setAccessible(true);
        return (Boolean) method.invoke(service, shape, batchId, responseQueue);
    }
}
