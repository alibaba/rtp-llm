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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.lang.reflect.Field;
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
 * Contract test for the per-rid prefill terminal event row
 * ({@code event=prefill_done}) written to engine_events.jsonl by the batch
 * completion callback in {@code FastRpcService.runPrefillBatch} — the
 * structured JSONL replacement of the former {@code mock_prefill_done}
 * stdout trace line.
 *
 * <p>The row is the engine-side half of the BIRTH-axis engine-exec
 * percentiles: aggregate_canvas_run.py joins {@code exec_ms} /
 * {@code prefill_done_ms} against the load client's {@code send_start_epoch_ms}
 * bucket by {@code rid} (the numeric GenerateInputPB.requestId, NOT the
 * trace-source rid string), so prefill exec lands on the same axis as
 * e2e/full_e2e. A schema drift here silently breaks that join, so the exact
 * JSON key set is asserted — if this test fails after an intentional schema
 * change, update the parser in aggregate_canvas_run.py in the same commit.
 *
 * <p>Prefill executes whole BATCHES, so the contract has two batch-shaped
 * properties beyond the schema: every member of one batch logs the SAME
 * exec_ms (the batch execution duration), the SAME prefill_done_ms (one
 * wall-clock stamp taken at batch completion) and the SAME batch_size. All
 * are asserted here via a single 3-request enqueueBatch.
 *
 * <p>Observation channel: the test injects its own response queues into the
 * service's {@code responseQueues} map (enqueueBatch's computeIfAbsent keeps
 * pre-registered entries), and the event row is written BEFORE the terminal
 * frame is offered on the same completion-callback iteration (no decode role
 * addr configured → startDecode returns false → finished=true frame offered
 * from the prefill side) — so each frame's arrival proves that member's row
 * was already written. The EngineEventLog autoflushes per row, so the file is
 * readable right after the frames land.
 *
 * <p>Regression guard: stdout must stay CLEAN of the legacy trace line —
 * per-request data flows exclusively through the JSONL file now.
 *
 * <p>The {@code cancelled=true} branch (a cancel marker landing before the
 * batch completes) is a race window that cannot be reproduced
 * deterministically; only the normal {@code cancelled=false} output is
 * asserted here.
 */
class PrefillDoneLogLineTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final int BASE_PORT = 63600;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;
    private Path eventsFile;

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
    void prefillBatchCompletionEmitsPerRidEventRow() throws Exception {
        // Flat 20 ms prefill formula at sleep_scale 0.1 → one batch books
        // ~2 ms regardless of batch size, so the 3-member batch completes in
        // milliseconds — comfortably observable, fast to wait.
        MockPerformanceModel model = MockEngineTestSupport.performanceModel(
                tempDir, "20", 0.1, 1000.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model);
        eventsFile = tempDir.resolve("engine_events.jsonl");
        prefill.setEngineEventLog(JavaMockEngineCluster.EngineEventLog.open(eventsFile.toString()));

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
            // One terminal frame per member proves its event row was written
            // (the JSONL write precedes the frame offer in the same callback
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

        // Regression guard: the per-request stdout trace line is gone — the
        // structured JSONL file is the only per-request engine-side output.
        String capturedLog = captured.toString(StandardCharsets.UTF_8);
        assertFalse(
                capturedLog.contains("mock_prefill_done"),
                "stdout must not carry the legacy mock_prefill_done trace line, got: " + capturedLog);

        List<JsonNode> rows = readEventRows();
        assertEquals(
                rids.length,
                rows.size(),
                "exactly one prefill_done event row per batch member");

        Map<Long, JsonNode> byRid = new LinkedHashMap<>();
        for (JsonNode row : rows) {
            assertEquals("prefill_done", row.path("event").asText(),
                    "row must carry event=prefill_done: " + row);
            assertTrue(row.has("rid") && row.get("rid").isIntegralNumber(),
                    "row must carry a numeric rid: " + row);
            // Exact schema contract — keep in sync with aggregate_canvas_run.py.
            for (String key : new String[] {
                    "event", "rid", "engine_name", "batch_id", "engine_arrival_ms",
                    "prefill_start_ms", "prefill_done_ms", "ttft_ms", "exec_ms",
                    "batch_size", "input_len", "cache_hit_tokens", "kv_used_tokens",
                    "cancelled"}) {
                assertTrue(row.has(key), "row must carry the key " + key + ": " + row);
            }
            byRid.put(row.get("rid").asLong(), row);
        }
        assertEquals(rids.length, byRid.size(), "each row must carry a distinct rid");

        long execMsFirst = -1L;
        long doneFirst = -1L;
        for (int i = 0; i < rids.length; i++) {
            JsonNode row = byRid.get(rids[i]);
            assertNotNull(row, "missing event row for rid " + rids[i]);

            long arrivalMs = row.get("engine_arrival_ms").asLong();
            long startMs = row.get("prefill_start_ms").asLong();
            long doneMs = row.get("prefill_done_ms").asLong();
            for (long stamp : new long[] {arrivalMs, startMs, doneMs}) {
                assertTrue(
                        stamp >= beforeMs - 1 && stamp <= afterMs + 1,
                        "epoch-ms stamps must be wall-clock timestamps inside the test"
                                + " window [" + beforeMs + ", " + afterMs + "], got " + stamp
                                + " (row: " + row + ")");
            }
            // System.currentTimeMillis() is NOT strictly monotonic on the
            // JVM (clock micro-adjustments can move a later sample back by
            // 1ms), so the causal ordering is asserted within 1ms of jitter.
            assertTrue(arrivalMs - 1 <= startMs && startMs - 1 <= doneMs,
                    "arrival <= start <= done must hold within 1ms clock jitter: " + row);
            assertEquals(
                    doneMs - arrivalMs,
                    row.get("ttft_ms").asLong(),
                    "ttft_ms must equal prefill_done_ms - engine_arrival_ms");

            long execMs = row.get("exec_ms").asLong();
            // exec_ms caliber: the BATCH execution duration (prefill runs
            // whole batches), i.e. formula "20" x sleep_scale 0.1 = 2 ms with
            // the scaledMs floor of 1. Assert the floor plus batch-uniformity
            // below; the exact model arithmetic is not the contract here.
            assertTrue(
                    execMs >= 1,
                    "exec_ms must be the batch execution duration (>= scaled floor), got "
                            + execMs);
            assertEquals(
                    3,
                    row.get("batch_size").asInt(),
                    "batch_size must echo the 3-member batch");
            if (i == 0) {
                execMsFirst = execMs;
                doneFirst = doneMs;
            } else {
                assertEquals(
                        execMsFirst,
                        execMs,
                        "members of one batch must log the SAME exec_ms (batch duration)");
                assertEquals(
                        doneFirst,
                        doneMs,
                        "members of one batch must log the SAME prefill_done_ms (one"
                                + " completion stamp)");
            }

            assertEquals(
                    inputLens[i],
                    row.get("input_len").asInt(),
                    "input_len must echo the request's input token count");
            assertEquals(false, row.get("cancelled").asBoolean(),
                    "normal completion must log cancelled=false");
        }
    }

    // ──────────── helpers ────────────

    private List<JsonNode> readEventRows() throws Exception {
        List<JsonNode> rows = new ArrayList<>();
        for (String line : Files.readAllLines(eventsFile, StandardCharsets.UTF_8)) {
            if (!line.isBlank()) {
                rows.add(MAPPER.readTree(line));
            }
        }
        return rows;
    }

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
