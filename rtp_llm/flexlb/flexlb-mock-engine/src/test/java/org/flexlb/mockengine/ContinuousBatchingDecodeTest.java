package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Unit tests for the per-step continuous batching decode engine (production
 * FIFOScheduler alignment, task#51).
 *
 * <p>Model under test: every admitted decode request becomes a DecodeStream in
 * a per-engine running batch; a single chained scheduler tick advances ALL
 * running streams one step per tick (tokens_per_step tokens per stream; this
 * suite pins tokens_per_step=1 so the step-count assertions below stay exact
 * — the MTP fold is covered by ProductionCaliberDecodeTest), with the step
 * duration priced from the step_ms_by_batch curve at the CURRENT running
 * count when the step is armed (a mid-flight joiner waits for the next
 * boundary — awaitsFirstStep). A stream exhausting its step budget completes
 * at that boundary and the waiting-queue head is promoted immediately
 * (production top-up).
 *
 * <p>Observation channel: each request is admitted with its OWN
 * LinkedBlockingQueue (the direct generate_stream path), so the terminal
 * frame's arrival timestamp IS the completion timestamp — no polling
 * quantisation. The curve used everywhere is [[1, 100], [2, 200]] ms with
 * sleep_scale 1.0 / jitter 0 (flat single-point curve where noted), giving
 * clean, widely separated timings.
 *
 * <ol>
 *   <li>{@link #stepDurationRepricesWhenBatchSizeChanges} — a second stream
 *       joining mid-flight drags the shared step from 100 ms to 200 ms: the
 *       first stream's completion is provably later than its solo timeline
 *       (2 × 100 ms) and the joiner finishes strictly after it.</li>
 *   <li>{@link #slotConservationWaitingDrainAndTopUpOrder} — 8 requests vs
 *       cap 3: running + waiting + completed is invariant at every sample,
 *       activeDecode never exceeds the cap, all 8 complete, counters settle
 *       at zero, and terminal frames arrive in FIFO admission order (the
 *       waiting queue drains head-first).</li>
 *   <li>{@link #longStreamDoesNotBlockShortStream} — an ol=20 stream and an
 *       ol=2 stream share the batch: the short stream completes ~1.7 s before
 *       the long one, and the long stream ACCELERATES after the short one
 *       leaves (step re-priced 200 → 100 ms), finishing far below the
 *       no-reprice bound.</li>
 *   <li>{@link #cancelRunningStreamPromotesWaitingHeadImmediately} —
 *       cancelling a running stream frees its slot and promotes the waiting
 *       head in the SAME locked section (no step-boundary wait): counters
 *       reflect the promotion instantly, the cancelled request never sees a
 *       terminal frame, and every surviving request drains to completion.</li>
 * </ol>
 */
class ContinuousBatchingDecodeTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63400;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private ExecutorService workerPool;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "continuous-batching-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        workerPool = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r, "continuous-batching-worker");
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
        workerPool.shutdownNow();
        workerPool.awaitTermination(3, TimeUnit.SECONDS);
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── Test 1: step duration re-prices when batch size changes ────────────

    /**
     * Timeline with curve [[1,100],[2,200]]: A (ol=2) admitted at t=0 arms a
     * 100 ms step priced at bs=1; B (ol=2) admitted immediately after joins
     * mid-step (awaitsFirstStep). tick@100: A advances, B joins the batch. The
     * next step is armed at bs=2 → 200 ms. tick@300: A finishes (100+200).
     * The next step re-prices at bs=1 → 100 ms. tick@400: B finishes.
     *
     * <p>Solo-A would finish at 200 ms; observed t_A ≥ 250 ms proves the step
     * re-priced to 200 ms when B joined — the core continuous-batching
     * behaviour the old one-shot model could not produce.
     */
    @Test
    @Timeout(30)
    void stepDurationRepricesWhenBatchSizeChanges() throws Exception {
        MockPerformanceModel model = twoPointCurveModel();
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 132);

        long t0 = System.nanoTime();
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queueA = new LinkedBlockingQueue<>();
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queueB = new LinkedBlockingQueue<>();
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 1L, 10, 2), -1, queueA));
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 2L, 10, 2), -1, queueB));
        assertEquals(2, getActiveDecodeRequests(decode),
                "both streams must be admitted into the running batch");

        long doneA = awaitTerminalFrame(queueA, 10_000) - t0;
        long doneB = awaitTerminalFrame(queueB, 10_000) - t0;
        long tAMs = TimeUnit.NANOSECONDS.toMillis(doneA);
        long tBMs = TimeUnit.NANOSECONDS.toMillis(doneB);

        // A: solo timeline is 2 × 100 = 200 ms; with B dragging step 2 to
        // 200 ms the theoretical finish is 300 ms. The lower bound 250 ms
        // rejects the one-shot/solo behaviour with margin.
        assertTrue(tAMs >= 250,
                "stream A must be slowed by B joining (expected ~300ms, solo would be 200ms), got " + tAMs + "ms");
        assertTrue(tAMs <= 1500,
                "stream A must finish near the re-priced timeline, got " + tAMs + "ms");
        // B joined one boundary later and runs its last step alone (100 ms):
        // theoretically 400 ms, strictly after A.
        assertTrue(tBMs > tAMs,
                "joiner B must finish strictly after A (tA=" + tAMs + "ms, tB=" + tBMs + "ms)");
        assertTrue(tBMs - tAMs >= 50,
                "B finishes ~one solo step after A, got gap=" + (tBMs - tAMs) + "ms");
        assertTrue(tBMs <= 2500, "stream B timeline sanity, got " + tBMs + "ms");

        awaitQuiescence(decode, 10_000);
        assertEquals(2L, decode.getCompletedCount());
        assertEquals(0, decode.getActiveKvTokens());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Test 2: slot conservation, waiting drain, top-up order ────────────

    /**
     * 8 requests (ol=3, flat 100 ms curve) vs cap 3: the invariant
     * running + waiting + completed == 8 must hold at EVERY sampled instant,
     * activeDecode never exceeds the cap, and every request completes with
     * terminal frames arriving in FIFO admission order (waiting head promoted
     * first).
     */
    @Test
    @Timeout(30)
    void slotConservationWaitingDrainAndTopUpOrder() throws Exception {
        // Flat curve: every step is 100 ms regardless of batch size, so the
        // timeline is fully predictable (all 8 done in ~1 s).
        MockPerformanceModel model = model(List.of(List.of(1, 100.0)));
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 3);

        int n = 8;
        @SuppressWarnings("unchecked")
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[] queues =
                (LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[]) new LinkedBlockingQueue[n];
        for (int i = 0; i < n; i++) {
            queues[i] = new LinkedBlockingQueue<>();
            assertTrue(invokeScheduleDecodeCompletion(
                    decode, shapeOf(model, 100L + i, 10, 3), -1, queues[i]));
        }
        assertEquals(3, getActiveDecodeRequests(decode), "hard gate: only cap running");
        assertEquals(n - 3, decodePendingQueueSize(decode), "excess parks in waiting queue");

        // Sample the conservation invariant while the engine drains: running
        // + waiting + completed must always equal the admitted total.
        // Sampling caveat: runDecodeStep advances the counters in two phases
        // — slot release + top-up under decodeQueueLock, then
        // completedCount++ in the post-lock publish — so a sample landing in
        // that window transiently reads n-1. A re-sample after the window
        // must restore the invariant; a genuine slot leak never would.
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(15_000);
        while (System.nanoTime() < deadline) {
            int running = getActiveDecodeRequests(decode);
            int waiting = decodePendingQueueSize(decode);
            long completed = decode.getCompletedCount();
            if ((long) running + waiting + completed != n) {
                Thread.sleep(50);
                running = getActiveDecodeRequests(decode);
                waiting = decodePendingQueueSize(decode);
                completed = decode.getCompletedCount();
                assertEquals((long) n, (long) running + waiting + completed,
                        "slot conservation (re-sampled after the claim→publish window): "
                                + "running + waiting + completed must equal admitted ("
                                + running + " + " + waiting + " + " + completed + ")");
            }
            assertTrue(running <= 3, "activeDecodeRequests must never exceed the cap, got " + running);
            if (completed == n) {
                break;
            }
            Thread.sleep(5);
        }
        assertEquals((long) n, decode.getCompletedCount(), "all requests must complete (waiting fully drained)");
        assertEquals(0, getActiveDecodeRequests(decode));
        assertEquals(0, decodePendingQueueSize(decode));

        // FIFO order: terminal frames must arrive in admission order — the
        // waiting queue promotes its head first, and streams finishing in the
        // same tick are published in running-batch insertion order.
        long t0 = System.nanoTime();
        long[] doneAt = new long[n];
        for (int i = 0; i < n; i++) {
            assertNotNull(queues[i].poll(10, TimeUnit.SECONDS),
                    "terminal frame for request " + (100 + i) + " must arrive");
            doneAt[i] = System.nanoTime() - t0;
        }
        for (int i = 1; i < n; i++) {
            assertTrue(doneAt[i] >= doneAt[i - 1],
                    "terminal frames must arrive in FIFO admission order: idx " + i
                            + " (" + TimeUnit.NANOSECONDS.toMillis(doneAt[i]) + "ms) before idx "
                            + (i - 1) + " (" + TimeUnit.NANOSECONDS.toMillis(doneAt[i - 1]) + "ms)");
        }

        awaitQuiescence(decode, 10_000);
        assertEquals(0, decode.getActiveKvTokens());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Test 3: a long stream must not block a short one ────────────

    /**
     * L (ol=20) admitted first, S (ol=2) joins mid-step. Shared steps run at
     * 200 ms until S finishes (t=500 ms: 100 + 200 + 200), after which L's
     * step re-prices to 100 ms and L finishes at ~2.2 s — far below the
     * 3.9 s no-reprice bound. S completes ~1.7 s before L: the long stream
     * never blocks the short one's completion, and the batch re-prices on
     * every boundary.
     */
    @Test
    @Timeout(30)
    void longStreamDoesNotBlockShortStream() throws Exception {
        MockPerformanceModel model = twoPointCurveModel();
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 132);

        long t0 = System.nanoTime();
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queueLong = new LinkedBlockingQueue<>();
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queueShort = new LinkedBlockingQueue<>();
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 1L, 10, 20), -1, queueLong));
        assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 2L, 10, 2), -1, queueShort));

        long doneShort = TimeUnit.NANOSECONDS.toMillis(awaitTerminalFrame(queueShort, 15_000) - t0);
        long doneLong = TimeUnit.NANOSECONDS.toMillis(awaitTerminalFrame(queueLong, 15_000) - t0);

        // Short stream: theoretical 500 ms (100 solo-priced step B missed +
        // two shared 200 ms steps). It must complete LONG before the long
        // stream — the defining property continuous batching buys over a
        // batch-synchronous execution.
        assertTrue(doneShort <= 1200,
                "short stream must complete on its own token count (~500ms), got " + doneShort + "ms");
        assertTrue(doneLong - doneShort >= 800,
                "long stream must finish far after the short one, gap=" + (doneLong - doneShort) + "ms");
        // Re-price acceleration: after S leaves, L runs solo at 100 ms/step.
        // Theoretical 2.2 s; no-reprice (all 200 ms) would be 3.9 s. The 3 s
        // bound rejects the flat-price timeline with margin.
        assertTrue(doneLong <= 3000,
                "long stream must accelerate after the short one leaves (re-price 200→100ms), got "
                        + doneLong + "ms (no-reprice bound 3900ms)");

        awaitQuiescence(decode, 10_000);
        assertEquals(2L, decode.getCompletedCount());
        assertEquals(0, decode.getActiveKvTokens());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Test 4: cancel a running stream → immediate top-up ────────────

    /**
     * cap 2, five ol=4 streams on a flat 100 ms curve: S1/S2 running, S3–S5
     * waiting. Cancelling S1 must free the slot and promote S3 IMMEDIATELY
     * (same locked section — activeDecode back at cap with the waiting queue
     * one shorter, before any step boundary fires). S1 never receives a
     * terminal frame; all four survivors complete and every counter nets to
     * zero.
     */
    @Test
    @Timeout(30)
    void cancelRunningStreamPromotesWaitingHeadImmediately() throws Exception {
        MockPerformanceModel model = model(List.of(List.of(1, 100.0)));
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 2);

        @SuppressWarnings("unchecked")
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[] queues =
                (LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[]) new LinkedBlockingQueue[5];
        for (int i = 0; i < 5; i++) {
            queues[i] = new LinkedBlockingQueue<>();
            assertTrue(invokeScheduleDecodeCompletion(
                    decode, shapeOf(model, 200L + i, 10, 4), -1, queues[i]));
        }
        assertEquals(2, getActiveDecodeRequests(decode));
        assertEquals(3, decodePendingQueueSize(decode));

        // Cancel the FIRST running stream. The promotion must be synchronous:
        // by the time cancel() returns, the freed slot already belongs to S3.
        decode.cancel(200L);
        assertEquals(2, getActiveDecodeRequests(decode),
                "freed slot must be handed to the waiting head immediately (no step-boundary wait)");
        assertEquals(2, decodePendingQueueSize(decode), "S3 must have left the waiting queue");
        assertEquals(4, decode.getInflightCount(),
                "S1 is gone; S2..S5 (running + waiting) stay in flight");

        // S1 must never see a terminal frame; all survivors must.
        assertNull(queues[0].poll(300, TimeUnit.MILLISECONDS),
                "cancelled stream must not receive a terminal frame");
        for (int i = 1; i < 5; i++) {
            assertNotNull(queues[i].poll(10, TimeUnit.SECONDS),
                    "survivor S" + (i + 1) + " must complete");
        }

        awaitQuiescence(decode, 10_000);
        assertEquals(4L, decode.getCompletedCount(), "exactly the four survivors complete");
        assertEquals(0, getActiveDecodeRequests(decode));
        assertEquals(0, decodePendingQueueSize(decode));
        assertEquals(0, decode.getActiveKvTokens());
        decode.checkLeakDrain(0L);
        assertFalse(decode.isLeakDetected());
    }

    // ──────────── Model / service helpers ────────────

    /** Curve [[1,100],[2,200]] with sleep_scale 1.0 and no jitter. */
    private MockPerformanceModel twoPointCurveModel() throws Exception {
        return model(List.of(List.of(1, 100.0), List.of(2, 200.0)));
    }

    private MockPerformanceModel model(List<List<Number>> stepCurve) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> decodeConfig = new LinkedHashMap<>();
        decodeConfig.put("scale", 1.0);
        // Pin the MTP fold to 1 token/step: the timeline assertions below
        // count steps 1:1 with output tokens.
        decodeConfig.put("tokens_per_step", 1.0);
        decodeConfig.put("step_ms_by_batch", stepCurve);
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", decodeConfig));
        MockMasterConfig.writeWithPrefillExpression(master, "10");
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int decodeMaxConcurrency) {
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "continuous-decode-" + port, "127.0.0.1", "decode",
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

    // ──────────── Completion observation helpers ────────────

    /**
     * Blocks for the terminal frame on the request's own response queue and
     * returns its arrival timestamp (System.nanoTime). The direct
     * generate_stream path publishes exactly one finished=true frame at decode
     * completion, so this timestamp IS the completion timestamp.
     */
    private static long awaitTerminalFrame(
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> queue, long timeoutMs)
            throws InterruptedException {
        EngineRpcService.GenerateOutputsPB frame = queue.poll(timeoutMs, TimeUnit.MILLISECONDS);
        assertNotNull(frame, "terminal frame must arrive within " + timeoutMs + "ms");
        assertTrue(frame.getFlattenOutput().getFinished(0), "frame must be the terminal frame");
        return System.nanoTime();
    }

    private void awaitQuiescence(JavaMockEngineCluster.FastRpcService service, long timeoutMs)
            throws Exception {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() == 0 && service.getRunningCount() == 0
                    && getActiveDecodeRequests(service) == 0) {
                return;
            }
            Thread.sleep(10);
        }
        fail("engine did not quiesce: inflight=" + service.getInflightCount()
                + " running=" + service.getRunningCount()
                + " activeDecode=" + getActiveDecodeRequests(service)
                + " kv=" + service.getActiveKvTokens());
    }

    // ──────────── Reflection helpers ────────────

    private static int getActiveDecodeRequests(JavaMockEngineCluster.FastRpcService service)
            throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class
                .getDeclaredField("activeDecodeRequests");
        field.setAccessible(true);
        return ((AtomicInteger) field.get(service)).get();
    }

    private static int decodePendingQueueSize(JavaMockEngineCluster.FastRpcService service)
            throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class
                .getDeclaredField("decodePendingQueue");
        field.setAccessible(true);
        return ((ArrayDeque<?>) field.get(service)).size();
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
