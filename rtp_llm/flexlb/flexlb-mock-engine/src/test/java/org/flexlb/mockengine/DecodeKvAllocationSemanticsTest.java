package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpGet;
import static org.flexlb.mockengine.MockEngineTestSupport.httpPost;
import static org.flexlb.mockengine.MockEngineTestSupport.httpPostResponse;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Decode KV allocation semantics (production-aligned 20260903, Zola C++
 * forensics) — the four gaps closed against the production behavior:
 *
 * <ol>
 *   <li><b>D-side ALLOCATE retry window</b> (DecodeRpcServer.cc:1190-1217
 *       EXECUTE_WITH_RETRY: {@code decode_retry_times=100} /
 *       {@code decode_retry_interval_ms=1} / {@code decode_retry_timeout_ms=100}):
 *       a failed reservation retries a FRESH acquire inside the window —
 *       another request's release flips a RETRYABLE verdict mid-window — and
 *       only the window-exhausted verdict terminates. The loop runs
 *       synchronously on the caller's enqueue thread (mock has no real gRPC
 *       async) and never holds the D-side cache monitor across the sleep.</li>
 *   <li><b>PERMANENT vs RETRYABLE classification</b>
 *       (KVCacheAllocator.cc:100-110): {@code required + reserve > physical
 *       total} = PERMANENT (retrying is pointless — production still spins it
 *       to the timeout; counted in {@code lack_mem_rejects}); pool total
 *       sufficient but temporarily short = RETRYABLE (counted in
 *       {@code kv_admission_fails}). Both families retry identically.</li>
 *   <li><b>reserve_step constant front-load window</b> (speculative
 *       {@code reserve_step_ = gen_num_per_circle + 1}, non-speculative = 0):
 *       decode demand prices {@code ceil((seq_len + reserve_step)/spb)} —
 *       initial admission front-loads the window, growth CONSUMES it
 *       (never re-appends); 0 keeps the exact fit bit-identical.</li>
 *   <li><b>Master-surface 8211</b> (the P engine's closeGrpcStream rewrite of
 *       a D-side RESOURCE_EXHAUSTED into DECODE_MALLOC_FAILED): decode-side
 *       reservation rejections ack 8211 with the raw 602 in the message
 *       text; prefill-pool rejections keep 602 (the two pools surface
 *       different codes, exactly like production).</li>
 * </ol>
 */
class DecodeKvAllocationSemanticsTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final int SPB = 1024;
    private static final int BASE_PORT = 64100;

    private static final Pattern PER_ENGINE_METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{engine_name=\"[^\"]+\",role=\"[^\"]+\",grpc_port=\"(\\d+)\",engine_ip=\"[^\"]+\"\\}\\s+(\\d+)");

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private MockControlServer controlServer;
    private int nextPortOffset;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "decode-kv-semantics-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        nextPortOffset = 0;
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ─────────────── Gap 1: the ALLOCATE retry window ───────────────

    /**
     * A RETRYABLE reservation failure flips inside the window: request A
     * pins 9 of the 10 D blocks (11 would be PERMANENT, 9 is temporarily
     * short), request B demands the same 9 — its P-side enqueue spins in the
     * D engine's ALLOCATE retry window until the background cancel releases
     * A's blocks, then the very next fresh acquire succeeds. The ack carries
     * NO error, neither failure counter ever ticks, and B's own reservation
     * holds 9 blocks (A's are gone).
     */
    @Test
    void retryWindowFlipsRetryableVerdictOnRelease() throws Exception {
        // Slow prefill (5000ms): A never reaches hand-off, so its D
        // reservation stays pinned in the reservation state machine until
        // the explicit cancel — the release path under test is deterministic.
        MockPerformanceModel model = performanceModel(tempDir, "5000", 1.0, 1.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 10);
        int decodePort = decode.getGrpcPort();

        // A (rid 100): 9 blocks fit outright (avail 10, reserve 1).
        assertEquals(1, enqueue(prefill, batch(1, slot(0,
                        inputWithDecode(100L, 9 * SPB, decodePort, 1))))
                .getSuccessesCount(),
                "A's reservation must succeed outright");
        assertEquals(9L, heldBlocks(decodePort), "A pins 9 of the 10 D blocks");

        // Widen the window for CI-load headroom (production 100/1/100 is
        // exercised by the exhaustion test below); the release lands at 50ms.
        decode.setDecodeAllocateRetryPolicy(500, 2, 2000);
        Thread releaser = new Thread(() -> {
            try {
                Thread.sleep(50);
                prefill.cancel(100L);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }, "kv-retry-window-releaser");
        releaser.start();

        long startNanos = System.nanoTime();
        // B (rid 101): same 9-block demand — RETRYABLE (9 + reserve 1 = 10 is
        // NOT > the physical 10), so it spins in the window until A releases.
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(2, slot(0, inputWithDecode(101L, 9 * SPB, decodePort, 1))));
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
        releaser.join(5000);

        assertEquals(0, ack.getErrorsCount(), "B must clear the window: " + ack);
        assertEquals(1, ack.getSuccessesCount(), "B is admitted after the flip");
        assertTrue(elapsedMs >= 50,
                "B must have WAITED for the release inside the window, took " + elapsedMs + "ms");
        assertEquals(9L, heldBlocks(decodePort),
                "B now holds the 9 blocks (A's are released)");
        Map<String, Map<Integer, Long>> metrics = perEngineMetrics();
        assertEquals(0L, metrics.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "a flipped verdict never counts");
        assertEquals(0L, metrics.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(decodePort, -1L),
                "a flipped verdict never counts (PERMANENT side)");
        assertFalse(prefill.isLeakDetected(), "no leak on P");
        assertFalse(decode.isLeakDetected(), "no leak on D");
    }

    /**
     * A window-exhausted RETRYABLE failure terminates: A pins 9 blocks for
     * the whole (default 100/1/100) window, B's fresh acquires keep failing
     * until the timeout, and the reject surfaces as the master-facing 8211
     * with the raw 602 in the message text — the RETRYABLE family counting
     * on the D engine's kv_admission_fails (lack_mem_rejects stays 0 on BOTH
     * engines: the P pool did not reject, and 9 + reserve fits the physical
     * total).
     */
    @Test
    void retryWindowExhaustionTerminatesWithMasterSurface8211() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "5000", 1.0, 1.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 10);
        int decodePort = decode.getGrpcPort();

        assertEquals(1, enqueue(prefill, batch(1, slot(0,
                        inputWithDecode(100L, 9 * SPB, decodePort, 1))))
                .getSuccessesCount(),
                "A's reservation must succeed outright");

        long startNanos = System.nanoTime();
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(2, slot(0, inputWithDecode(101L, 9 * SPB, decodePort, 1))));
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);

        assertEquals(1, ack.getErrorsCount(), "B must be rejected after the window");
        assertEquals(0, ack.getSuccessesCount(), "a rejected request is never admitted");
        assertEquals(JavaMockEngineCluster.DECODE_LACK_MEM_ERROR_CODE,
                ack.getErrors(0).getErrorInfo().getErrorCode(),
                "the decode-side reject acks the master-surface 8211");
        assertTrue(ack.getErrors(0).getErrorInfo().getErrorMessage().contains("602"),
                "the raw 602 travels in the message text: "
                        + ack.getErrors(0).getErrorInfo().getErrorMessage());
        assertTrue(ack.getErrors(0).getErrorInfo().getErrorMessage()
                        .contains("ALLOCATE retry window"),
                "the message must name the exhausted window: "
                        + ack.getErrors(0).getErrorInfo().getErrorMessage());
        assertTrue(elapsedMs >= 90,
                "the default 100ms window must actually be spun, took " + elapsedMs + "ms");

        Map<String, Map<Integer, Long>> metrics = perEngineMetrics();
        assertEquals(1L, metrics.get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "the RETRYABLE exhaustion counts on the D engine");
        assertEquals(0L, metrics.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(decodePort, -1L),
                "9 + reserve fits the physical total — NOT PERMANENT");
        assertEquals(0L, metrics.get("mock_engine_lack_mem_rejects_total")
                        .getOrDefault(prefill.getGrpcPort(), -1L),
                "the P pool did not reject (100-block pool, 9-block demand)");
        assertEquals(9L, heldBlocks(decodePort),
                "A's pin survives the whole episode");
        assertFalse(prefill.isLeakDetected(), "no leak on P");
        assertFalse(decode.isLeakDetected(), "no leak on D");
    }

    /**
     * /set_perf {@code decode_retry_times=0} short-circuits the window: the
     * same RETRYABLE geometry fails IMMEDIATELY (elapsed stays far below the
     * 100ms production window) — the runtime override is wired end to end.
     * A negative value is a 400 (neither a budget nor the explicit zero).
     */
    @Test
    void setPerfRetryPolicyShortCircuitsTheWindow() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "5000", 1.0, 1.0);
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(model, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(model, 10);
        int decodePort = decode.getGrpcPort();

        httpPost(controlPort(), "/set_perf", String.format(
                "{\"engine\":\"%s\",\"decode_retry_times\":0,"
                        + "\"decode_retry_interval_ms\":1,\"decode_retry_timeout_ms\":100}",
                decode.getEngineName()));

        assertEquals(1, enqueue(prefill, batch(1, slot(0,
                        inputWithDecode(100L, 9 * SPB, decodePort, 1))))
                .getSuccessesCount(),
                "A's reservation must succeed outright");

        long startNanos = System.nanoTime();
        EngineRpcService.EnqueueBatchResponsePB ack =
                enqueue(prefill, batch(2, slot(0, inputWithDecode(101L, 9 * SPB, decodePort, 1))));
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);

        assertEquals(1, ack.getErrorsCount(), "B still fails — zero retries configured");
        assertEquals(JavaMockEngineCluster.DECODE_LACK_MEM_ERROR_CODE,
                ack.getErrors(0).getErrorInfo().getErrorCode());
        assertTrue(elapsedMs < 90,
                "zero retries must fail fast, took " + elapsedMs + "ms");
        assertEquals(1L, perEngineMetrics().get("mock_engine_kv_admission_fails_total")
                        .getOrDefault(decodePort, -1L),
                "the RETRYABLE failure still counts on the D engine");

        HttpResponse<String> rejected = httpPostResponse(controlPort(), "/set_perf", String.format(
                "{\"engine\":\"%s\",\"decode_retry_times\":-1}", decode.getEngineName()));
        assertEquals(400, rejected.statusCode(), "negative decode_retry_times must be a 400");
        assertTrue(MAPPER.readTree(rejected.body()).get("error").asText()
                        .contains("decode_retry"),
                "the error should name the field, got: " + rejected.body());
    }

    // ─────────────── Gap 2: PERMANENT vs RETRYABLE classification ───────────────

    /**
     * Pool-unit classification (production KVCacheAllocator.cc:100-110): with
     * a 10-block pool (reserve = 1), a 9-block demand against 1 available
     * block is RETRYABLE (9 + 1 is NOT > the physical 10 — a release can
     * flip it); an 11-block demand is PERMANENT (11 + 1 > 10 — the request
     * can never fit, even on an EMPTY pool). Both calibers (acquire /
     * acquireWithReuse) classify identically; failures change no state.
     */
    @Test
    void permanentVsRetryableClassification() {
        MockLruBlockCache cache = new MockLruBlockCache(10); // reserve = ceil(0.05 * 10) = 1
        MockLruBlockCache.BlockLease pin = cache.acquire(9, List.of());
        assertNotNull(pin, "9 blocks fit the empty 10-block pool (avail − need = reserve)");

        assertEquals(MockLruBlockCache.AllocationFailure.RETRYABLE,
                cache.acquireDetailed(9, List.of()).failure(),
                "total sufficient, temporarily short → RETRYABLE");
        assertEquals(MockLruBlockCache.AllocationFailure.RETRYABLE,
                cache.acquireWithReuseDetailed(9, List.of()).failure(),
                "the decode caliber classifies the same geometry identically");
        assertEquals(MockLruBlockCache.AllocationFailure.PERMANENT,
                cache.acquireDetailed(11, List.of()).failure(),
                "required + reserve > physical total → PERMANENT");
        assertEquals(MockLruBlockCache.AllocationFailure.PERMANENT,
                cache.acquireWithReuseDetailed(11, List.of()).failure(),
                "the decode caliber classifies PERMANENT identically");
        assertNull(cache.acquireDetailed(9, List.of()).lease(),
                "a failed attempt yields no lease");

        cache.release(pin);
        assertEquals(MockLruBlockCache.AllocationFailure.PERMANENT,
                cache.acquireDetailed(11, List.of()).failure(),
                "PERMANENT holds even on the recovered pool — the request never fits");

        // Growth twin: growing toward 11 total on a 10-block pool is
        // PERMANENT; growing toward 9 while 9 are already held is a no-op.
        MockLruBlockCache.BlockLease lease = cache.acquire(9, List.of());
        assertNotNull(lease);
        assertEquals(MockLruBlockCache.AllocationFailure.PERMANENT,
                cache.growTo(lease, 11).failure(),
                "growth past the physical total classifies PERMANENT");
        assertTrue(cache.growTo(lease, 9).success(),
                "already-satisfied growth is a no-op success");
    }

    // ─────────────── Gap 3: the reserve_step front-load window ───────────────

    /**
     * decode.reserve_step front-loads a CONSTANT look-ahead: with step=1 an
     * exactly-8-block input prices 9 (the initial admission holds one extra
     * block); with step=0 (the non-speculative default) the same input
     * prices exactly 8 — bit-identical to the pre-gap behavior. The demand
     * caliber is shared by the initial admission and every growth step
     * (decodeDemandBlocks), so the window is CONSUMED by seq_len growth,
     * never re-appended.
     */
    @Test
    void reserveStepFrontLoadsDecodeDemand() throws Exception {
        MockPerformanceModel speculative = performanceModel(
                tempDir, "5000", 1.0, 1.0, Map.of(), Map.of("reserve_step", 1));
        JavaMockEngineCluster.FastRpcService prefill = newPrefillService(speculative, 100);
        JavaMockEngineCluster.FastRpcService decode = newDecodeService(speculative, 10);
        int decodePort = decode.getGrpcPort();

        assertEquals(9, decode.decodeDemandBlocks(8 * SPB),
                "step=1 front-loads ceil((8192 + 1)/1024) = 9 blocks");
        assertEquals(8, decode.decodeDemandBlocks(8 * SPB - 1),
                "the window only trips at the boundary (8191 + 1 still prices 8)");
        assertEquals(9, decode.decodeDemandBlocks(8 * SPB + 1),
                "growth at 8193 consumes the SAME window (still 9, not 10)");

        // End to end: the speculative reservation holds the front-loaded 9.
        assertEquals(1, enqueue(prefill, batch(1, slot(0,
                        inputWithDecode(100L, 8 * SPB, decodePort, 1))))
                .getSuccessesCount());
        assertEquals(9L, heldBlocks(decodePort),
                "the initial admission holds the front-loaded block");

        // Control: the default (non-speculative) model keeps the exact fit.
        MockPerformanceModel plain = performanceModel(tempDir, "5000", 1.0, 1.0);
        JavaMockEngineCluster.FastRpcService prefill0 = newPrefillService(plain, 100);
        JavaMockEngineCluster.FastRpcService decode0 = newDecodeService(plain, 10);
        int decode0Port = decode0.getGrpcPort();
        assertEquals(8, decode0.decodeDemandBlocks(8 * SPB),
                "step=0 (default) keeps ceil(8192/1024) = 8 exactly");
        assertEquals(1, enqueue(prefill0, batch(1, slot(0,
                        inputWithDecode(200L, 8 * SPB, decode0Port, 1))))
                .getSuccessesCount());
        assertEquals(8L, heldBlocks(decode0Port),
                "no front-load without reserve_step");
        assertFalse(decode.isLeakDetected(), "no leak on the speculative D");
        assertFalse(decode0.isLeakDetected(), "no leak on the plain D");
    }

    // ────────────────── helpers ──────────────────

    private JavaMockEngineCluster.FastRpcService newPrefillService(
            MockPerformanceModel model, int blocks) {
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService service =
                new JavaMockEngineCluster.FastRpcService(
                        "prefill",
                        EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                        port,
                        services,
                        scheduler,
                        model,
                        blocks,
                        new JavaMockEngineCluster.ClusterStats());
        services.put(port, service);
        startControlServer();
        return service;
    }

    private JavaMockEngineCluster.FastRpcService newDecodeService(
            MockPerformanceModel model, int blocks) {
        int port = BASE_PORT + nextPortOffset++;
        JavaMockEngineCluster.FastRpcService service =
                new JavaMockEngineCluster.FastRpcService(
                        "decode-" + port,
                        "127.0.0.1",
                        "decode",
                        EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                        port,
                        services,
                        scheduler,
                        model,
                        blocks,
                        new JavaMockEngineCluster.ClusterStats(),
                        10_000_000L,
                        8);
        services.put(port, service);
        startControlServer();
        return service;
    }

    private void startControlServer() {
        if (controlServer == null) {
            try {
                controlServer = new MockControlServer(
                        services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
                controlServer.start();
            } catch (IOException e) {
                throw new IllegalStateException("control server failed to start", e);
            }
        }
    }

    private int controlPort() {
        assertNotNull(controlServer, "control server must be running");
        return controlServer.getPort();
    }

    private Map<String, Map<Integer, Long>> perEngineMetrics()
            throws IOException, InterruptedException {
        Map<String, Map<Integer, Long>> result = new java.util.HashMap<>();
        Matcher matcher = PER_ENGINE_METRIC_PATTERN.matcher(
                httpGet(controlPort(), "/metrics?per_engine=true"));
        while (matcher.find()) {
            result.computeIfAbsent(matcher.group(1), k -> new java.util.HashMap<>())
                    .put(Integer.parseInt(matcher.group(2)), Long.parseLong(matcher.group(3)));
        }
        return result;
    }

    private long heldBlocks(int port) throws IOException, InterruptedException {
        return perEngineMetrics().get("mock_engine_held_blocks")
                .getOrDefault(port, -1L);
    }
}
