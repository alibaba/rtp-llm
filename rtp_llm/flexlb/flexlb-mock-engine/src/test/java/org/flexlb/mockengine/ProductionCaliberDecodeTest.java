package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Pins the task #69 production-caliber decode model: per-step pricing with the
 * production DSv4 linear fit (step_ms = 19.5 + 0.175 × running) and the MTP
 * acceptance fold (2.6 tokens/step), plus the config-surface contracts.
 *
 * <p>Production anchors (task #68 measurements, /tmp/km68 buckets):
 * <ul>
 *   <li>step_ms = 20.15 + 0.174 × batch (R² = 0.82); TPOT = 8.33 + 0.0644 × batch (R² = 0.90)</li>
 *   <li>MTP accept 2.54–2.88 tokens/step (slightly lower at full batch)</li>
 *   <li>engine throughput: low batch (running ≈ 4) ≈ 519 tok/s, full batch (128) ≈ 7726 tok/s</li>
 * </ul>
 *
 * <p>The drain-rate assertions run ONE decode engine at a steady running count
 * and compare achieved tokens/s against those production anchors with a ±15%
 * band (scheduler-timer quantisation + the one extra ramp-up step at admission
 * account for a few percent; the band rejects the removed per-token caliber,
 * which would sit ~2.8×–5.5× off).
 */
class ProductionCaliberDecodeTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63800;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(8, runnable -> {
            Thread thread = new Thread(runnable, "production-caliber-scheduler");
            thread.setDaemon(true);
            return thread;
        });
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── config surface ────────────

    @Test
    void defaultsAreTheProductionLinearFitWithMtpFold() throws Exception {
        MockPerformanceModel model = model(Map.of()); // no decode declaration at all
        assertEquals(2.6, model.tokensPerStep(), 1e-9, "MTP fold default");
        // step 19.5 + 0.175 × running, rounded: 1→19.675→20, 4→20.2→20, 128→41.9→42
        assertEquals(20L, model.decodeStepDelayMs(1));
        assertEquals(20L, model.decodeStepDelayMs(4));
        assertEquals(42L, model.decodeStepDelayMs(128));
        // MTP fold arithmetic: ceil(outputLen / 2.6)
        assertEquals(1, model.decodeSteps(1));
        assertEquals(10, model.decodeSteps(26));
        assertEquals(11, model.decodeSteps(27));
        assertEquals(193, model.decodeSteps(500));
        // External semantics kept: decodeMs = steps × step × scale ("duration of
        // outputLen tokens"). 26 tokens solo = 10 steps × 19.675 = 196.75 → 197;
        // 27 tokens needs 11 steps → 216.
        assertEquals(197L, model.decodeMs(26, 1));
        assertEquals(216L, model.decodeMs(27, 1));
    }

    @Test
    void perTokenMsIsRemovedAndFailsFast() {
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> model(Map.of("per_token_ms", 45.0)));
        assertTrue(ex.getMessage().contains("per_token_ms is removed"),
                "migration hint must name per_token_ms: " + ex.getMessage());
        assertTrue(ex.getMessage().contains("per STEP"),
                "migration hint must point at the per-step caliber: " + ex.getMessage());
    }

    @Test
    void curveAndLinearCoefficientsAreMutuallyExclusive() {
        Map<String, Object> conflicting = new LinkedHashMap<>();
        conflicting.put("step_ms_by_batch", List.of(List.of(1, 5.0)));
        conflicting.put("step_base_ms", 10.0);
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> model(conflicting));
        assertTrue(ex.getMessage().contains("mutually exclusive"),
                "conflict must be explicit: " + ex.getMessage());
    }

    @Test
    void invalidTokensPerStepFailsFast() {
        IllegalStateException ex = assertThrows(IllegalStateException.class,
                () -> model(Map.of("tokens_per_step", 0.0)));
        assertTrue(ex.getMessage().contains("tokens_per_step must be > 0"),
                "reason must be named: " + ex.getMessage());
    }

    @Test
    void explicitCoefficientsOverrideTheFitAndTheCurveBeatsTheLinearDefault() throws Exception {
        MockPerformanceModel linear = model(Map.of("step_base_ms", 10.0, "step_per_running_ms", 1.0));
        assertEquals(15L, linear.decodeStepDelayMs(5), "10 + 1×5");
        MockPerformanceModel curve = model(Map.of("step_ms_by_batch", List.of(List.of(1, 5.0))));
        assertEquals(5L, curve.decodeStepDelayMs(1), "declared curve wins over the linear default");
        // tokens_per_step stays global across both step-latency sources.
        MockPerformanceModel curveFold = model(Map.of(
                "tokens_per_step", 4.0, "step_ms_by_batch", List.of(List.of(1, 5.0))));
        assertEquals(3, curveFold.decodeSteps(10), "ceil(10/4)");
        assertEquals(2.6, model(Map.of()).tokensPerStep(), 1e-9);
    }

    // ──────────── production-anchor drain rates ────────────

    /**
     * Low-batch anchor: running = 4 steady streams on one engine. Production
     * ≈ 519 tok/s (task #68 low-batch bucket); the model predicts
     * 4 × 2.6 / (19.5 + 0.175×4) = 514.7 tok/s.
     */
    @Test
    @Timeout(60)
    void lowBatchDrainRateMatchesProductionAnchor() throws Exception {
        double tokPerSec = drainRate(4, 500);
        // Production anchor 519; model 514.7. ±15% band.
        assertWithinBand(tokPerSec, 519.0, 0.15, "low batch (running=4)");
    }

    /**
     * Full-batch anchor: running = 128 (the engine decode concurrency cap,
     * matching production). Production ≈ 7726 tok/s; the model predicts
     * 128 × 2.6 / (19.5 + 0.175×128) = 7943 tok/s.
     */
    @Test
    @Timeout(60)
    void fullBatchDrainRateMatchesProductionAnchor() throws Exception {
        double tokPerSec = drainRate(128, 100);
        // Production anchor 7726; model 7943 (+2.8%). ±15% band.
        assertWithinBand(tokPerSec, 7726.0, 0.15, "full batch (running=128)");
    }

    /**
     * Single-stream sanity (informational anchor): production TPOT at bs=1 is
     * 8.39 ms/token ≈ 119 tok/s; the model gives 2.6/19.675 = 132 tok/s
     * (+11%). The old fixed per_token_ms=45 caliber sat at 22 tok/s — 5.5× off.
     */
    @Test
    @Timeout(60)
    void singleStreamDrainRateMatchesProductionTpotCaliber() throws Exception {
        double tokPerSec = drainRate(1, 500);
        assertWithinBand(tokPerSec, 119.0, 0.20, "single stream (running=1)");
    }

    // ──────────── helpers ────────────

    /** Load a model whose "decode" section is {@code extra} over the base. */
    private MockPerformanceModel model(Map<String, Object> decodeExtra) throws IOException {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> decode = new LinkedHashMap<>();
        decode.put("scale", 1.0);
        decode.putAll(decodeExtra);
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", decode));
        MockMasterConfig.writeWithPrefillExpression(master, "10");
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    /**
     * Drains {@code streams} concurrent requests of {@code outputLen} tokens on
     * ONE decode engine at the production defaults and returns the achieved
     * tokens/s (total tokens / wall-clock from first admission to the last
     * terminal frame).
     */
    private double drainRate(int streams, int outputLen) throws Exception {
        MockPerformanceModel model = model(Map.of());
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService decode = new JavaMockEngineCluster.FastRpcService(
                "production-caliber-" + port, "127.0.0.1", "decode",
                EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                port, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats(),
                10_000_000L, 128);
        services.put(port, decode);

        @SuppressWarnings("unchecked")
        LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[] queues =
                (LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>[]) new LinkedBlockingQueue[streams];
        long start = System.nanoTime();
        for (int i = 0; i < streams; i++) {
            queues[i] = new LinkedBlockingQueue<>();
            assertTrue(invokeScheduleDecodeCompletion(decode, shapeOf(model, 1_000L + i, 10, outputLen), -1, queues[i]),
                    "stream " + i + " must be admitted (cap 128)");
        }
        long tokens = (long) streams * outputLen;
        for (int i = 0; i < streams; i++) {
            EngineRpcService.GenerateOutputsPB frame = queues[i].poll(30, TimeUnit.SECONDS);
            assertTrue(frame != null && frame.getFlattenOutput().getFinished(0),
                    "terminal frame for stream " + i + " must arrive");
        }
        double elapsedSec = (System.nanoTime() - start) / 1e9;
        double tokPerSec = tokens / elapsedSec;
        System.out.printf("[%d streams x %d tokens] %.0f tok/s in %.2fs%n",
                streams, outputLen, tokPerSec, elapsedSec);
        decode.checkLeakDrain(0L);
        return tokPerSec;
    }

    private static void assertWithinBand(double actual, double anchor, double band, String label) {
        assertTrue(actual >= anchor * (1 - band) && actual <= anchor * (1 + band),
                String.format("%s: %.0f tok/s outside production anchor %.0f ±%.0f%%",
                        label, actual, anchor, band * 100));
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
