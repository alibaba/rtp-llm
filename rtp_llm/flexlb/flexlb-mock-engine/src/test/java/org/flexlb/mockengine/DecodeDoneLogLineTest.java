package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Contract test for the per-rid decode terminal event row
 * ({@code event=decode_done}) written to engine_events.jsonl by
 * {@code FastRpcService.publishDecodeCompletion} — the structured JSONL
 * replacement of the former {@code mock_decode_done} stdout trace line.
 *
 * <p>The row is the engine-side half of the schedule-only full-e2e metric:
 * aggregate_canvas_run.py joins {@code decode_done_ms} / {@code exec_ms}
 * against the load client's {@code send_start_epoch_ms} by {@code rid} (the
 * numeric GenerateInputPB.requestId, NOT the trace-source rid string). A
 * schema drift here silently breaks that join, so the exact JSON key set is
 * asserted — if this test fails after an intentional schema change, update
 * the parser in aggregate_canvas_run.py in the same commit.
 *
 * <p>Observation channel: each request is admitted through the reflection
 * admission point with its OWN response queue (direct generate_stream path),
 * and the event row is written BEFORE the terminal frame is offered on the
 * same thread — so the frame's arrival proves the row was already written
 * (the EngineEventLog autoflushes per row).
 *
 * <p>Regression guard: stdout must stay CLEAN of the legacy trace line —
 * per-request data flows exclusively through the JSONL file now.
 *
 * <p>The {@code cancelled=true} branch (a cancel marker landing between the
 * terminal claim and the publish) is a narrow race window that cannot be
 * reproduced deterministically; only the normal {@code cancelled=false}
 * output is asserted here.
 */
class DecodeDoneLogLineTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

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
    void decodeCompletionEmitsPerRidEventRow() throws Exception {
        // Flat 20 ms model steps at sleep_scale 0.1 (decodeModel default):
        // each step books/sleeps ~2 ms, so every stream finishes in tens of
        // ms — comfortably observable, fast to wait.
        MockPerformanceModel model = MockEngineTestSupport.decodeModel(tempDir, 20.0, null);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 8);
        Path eventsFile = tempDir.resolve("engine_events.jsonl");
        decode.setEngineEventLog(JavaMockEngineCluster.EngineEventLog.open(eventsFile.toString()));
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

        // Regression guard: the per-request stdout trace line is gone — the
        // structured JSONL file is the only per-request engine-side output.
        String capturedLog = captured.toString(StandardCharsets.UTF_8);
        assertFalse(
                capturedLog.contains("mock_decode_done"),
                "stdout must not carry the legacy mock_decode_done trace line, got: " + capturedLog);

        List<JsonNode> rows = new ArrayList<>();
        for (String line : Files.readAllLines(eventsFile, StandardCharsets.UTF_8)) {
            if (!line.isBlank()) {
                rows.add(MAPPER.readTree(line));
            }
        }
        assertEquals(3, rows.size(),
                "exactly one decode_done event row per completed stream");

        Map<Long, JsonNode> byRid = new LinkedHashMap<>();
        for (JsonNode row : rows) {
            assertEquals("decode_done", row.path("event").asText(),
                    "row must carry event=decode_done: " + row);
            // Exact schema contract — keep in sync with aggregate_canvas_run.py.
            for (String key : new String[] {
                    "event", "rid", "engine_name", "batch_id", "engine_arrival_ms",
                    "decode_start_ms", "decode_done_ms", "exec_ms", "batch_size",
                    "output_len", "kv_used_tokens", "cancelled"}) {
                assertTrue(row.has(key), "row must carry the key " + key + ": " + row);
            }
            byRid.put(row.get("rid").asLong(), row);
        }
        assertEquals(3, byRid.size(), "each row must carry a distinct rid");
        long[] expectedRids = {4242L, 4243L, 4244L};
        int[] expectedOutputLens = {4, 6, 2};
        for (int i = 0; i < expectedRids.length; i++) {
            JsonNode row = byRid.get(expectedRids[i]);
            assertNotNull(row, "missing event row for rid " + expectedRids[i]);

            long arrivalMs = row.get("engine_arrival_ms").asLong();
            long startMs = row.get("decode_start_ms").asLong();
            long doneMs = row.get("decode_done_ms").asLong();
            for (long stamp : new long[] {arrivalMs, startMs, doneMs}) {
                assertTrue(stamp >= beforeMs - 1 && stamp <= afterMs + 1,
                        "epoch-ms stamps must be wall-clock timestamps inside the test"
                                + " window [" + beforeMs + ", " + afterMs + "], got " + stamp
                                + " (row: " + row + ")");
            }
            assertTrue(arrivalMs <= startMs && startMs <= doneMs,
                    "arrival <= start <= done must hold: " + row);

            long execMs = row.get("exec_ms").asLong();
            // exec_ms caliber: the summed BOOKED step durations — each step
            // prices at model step_ms × sleep_scale (decodeModel() runs
            // sleep_scale=0.1, so one 20ms step books 2ms), and steps fold
            // tokens_per_step tokens (MTP), e.g. output_len=4 → 2 steps × 2ms
            // = 4ms. Assert only the one-booked-step lower bound; the exact
            // sum depends on tokens_per_step and is not the contract here.
            assertTrue(execMs >= 2,
                    "exec_ms must be the summed booked step durations "
                            + "(>= one 20ms*0.1 step), got " + execMs);

            assertTrue(row.get("batch_size").asInt() >= 1,
                    "batch_size must count this stream at the terminal step, got "
                            + row.get("batch_size"));

            assertEquals(expectedOutputLens[i], row.get("output_len").asInt(),
                    "output_len must echo the stream's output token budget");
            assertEquals(false, row.get("cancelled").asBoolean(),
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
