package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Targeted regression tests for CodeReview fixes in the Java mock engine.
 *
 * <p>Each test directly covers one of the issues identified during code review,
 * ensuring the fixes are not regressed:
 *
 * <ol>
 *   <li>{@link #cancelOnDecodeEngineDecrementsAllCounters} — cancel() on a decode
 *       engine must decrement activeDecodeRequests and activeKvTokens in addition
 *       to pendingRequests.</li>
 *   <li>{@link #cancelledRequestNotForwardedToDecode} — a request cancelled on the
 *       prefill engine before prefill completes must not be forwarded to the decode
 *       engine (prefill completion callback checks cancelledRequests).</li>
 *   <li>{@link #putIfAbsentAtomicUnderConcurrency} — scheduleDecodeCompletion uses
 *       putIfAbsent so that concurrent calls for the same requestId do not
 *       double-count any counter.</li>
 * </ol>
 */
class CodeReviewFixTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 62900;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private ExecutorService workerPool;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(8, runnable -> {
            Thread thread = new Thread(runnable, "codereview-fix-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        workerPool = Executors.newCachedThreadPool(r -> {
            Thread thread = new Thread(r, "codereview-fix-worker");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
        }
        if (workerPool != null) {
            workerPool.shutdownNow();
            workerPool.awaitTermination(3, TimeUnit.SECONDS);
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    // ──────────── Test 1: cancel() on decode engine decrements all counters ────────────

    /**
     * Verifies that cancel() on a decode engine decrements activeDecodeRequests,
     * activeKvTokens, and pendingRequests — not just pendingRequests alone.
     *
     * <p>The fix added the decode-specific counter decrements inside the
     * {@code if (roleType == ROLE_TYPE_DECODE)} branch of cancel().
     */
    @Test
    void cancelOnDecodeEngineDecrementsAllCounters() throws Exception {
        // decode step 10000 × sleep_scale 0.1 = 1000 ms, giving ample time to
        // inspect counters and cancel before the scheduled completion fires.
        MockPerformanceModel model = model("10", 10000.0);
        startCluster(model, 1, 1);

        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);
        long requestId = 42L;
        int inputLen = 10;
        // Block-pool caliber: inputLen=10 rounds up to ceil(10/1024)=1 block,
        // so the lease pins 1 x spb = 1024 tokens.
        long expectedKvTokens = 1024L;
        MockPerformanceModel.RequestShape shape = shapeOf(model, requestId, inputLen);

        // Directly invoke the private scheduleDecodeCompletion via reflection.
        invokeScheduleDecodeCompletion(decode, shape, -1, null);

        // After scheduling, all three counters must reflect the single request.
        assertEquals(1, getActiveDecodeRequests(decode),
                "activeDecodeRequests should be 1 after scheduling");
        assertEquals(expectedKvTokens, decode.getActiveKvTokens(),
                "activeKvTokens should be " + expectedKvTokens + " after scheduling");
        assertEquals(1, decode.getInflightCount(),
                "pendingRequests should be 1 after scheduling");
        assertEquals(1, decode.getRunningCount(),
                "runningTasks should contain 1 entry after scheduling");

        // Cancel the request mid-flight.
        decode.cancel(requestId);

        // All counters must now be zero — this is the core of the fix.
        assertEquals(0, getActiveDecodeRequests(decode),
                "activeDecodeRequests should be 0 after cancel");
        assertEquals(0, decode.getActiveKvTokens(),
                "activeKvTokens should be 0 after cancel");
        assertEquals(0, decode.getInflightCount(),
                "pendingRequests should be 0 after cancel");
        assertEquals(0, decode.getRunningCount(),
                "runningTasks should be empty after cancel");
        assertFalse(decode.isLeakDetected(),
                "no leak should be detected after cancel");
    }

    // ──────────── Test 2: cancelled request not forwarded to decode ────────────

    /**
     * Verifies that a request cancelled on the prefill engine before prefill
     * completes is not forwarded to the decode engine.
     *
     * <p>The fix added a {@code cancelledRequests.contains(requestId)} check in
     * the prefill completion callback, skipping {@code startDecode} when the
     * request has already been cancelled.
     */
    @Test
    void cancelledRequestNotForwardedToDecode() throws Exception {
        // prefill formula "500" × sleep_scale 0.1 = 50 ms prefill, giving a
        // window to cancel before the prefill completion callback fires.
        MockPerformanceModel model = model("500", 1.0);
        startCluster(model, 1, 1);

        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);
        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);
        long requestId = 7L;
        int decodePort = decode.getGrpcPort();

        // Enqueue a single request with decode routing.
        EngineRpcService.GenerateInputPB input = inputWithDecode(requestId, 10, decodePort);
        EngineRpcService.EnqueueBatchResponsePB response =
                enqueue(prefill, batch(9000, slot(0, input)));
        assertEquals(0, response.getErrorsCount(), "enqueue should have 0 errors");
        assertEquals(1, response.getSuccessesCount(), "enqueue should have 1 success");

        // Cancel immediately while prefill is still in-flight (50 ms window).
        prefill.cancel(requestId);

        // Wait for the prefill completion callback to fire and drain inflight.
        awaitAllInflightZero(5_000);

        // The decode engine must not have received the request.
        assertEquals(0, decode.getInflightCount(),
                "decode pendingRequests should be 0 — request must not be forwarded");
        assertEquals(0, decode.getRunningCount(),
                "decode runningTasks should be empty — request must not be forwarded");
        assertEquals(0, decode.getActiveKvTokens(),
                "decode activeKvTokens should be 0 — request must not be forwarded");
        assertEquals(0, decode.getAcceptedCount(),
                "decode acceptedCount should be 0 — request must not be forwarded");

        // Prefill engine should also be clean.
        assertEquals(0, prefill.getInflightCount(),
                "prefill pendingRequests should be 0 after completion");
        assertEquals(0, prefill.getRunningCount(),
                "prefill runningTasks should be empty after completion");
        assertFalse(prefill.isLeakDetected(),
                "no leak should be detected on prefill engine");
        assertFalse(decode.isLeakDetected(),
                "no leak should be detected on decode engine");
    }

    // ──────────── Test 3: putIfAbsent atomic under concurrency ────────────

    /**
     * Verifies that scheduleDecodeCompletion uses putIfAbsent atomically, so
     * that concurrent calls for the same requestId do not double-count any
     * counter.
     *
     * <p>The fix replaced a {@code containsKey + put} sequence with
     * {@code putIfAbsent}, closing the race window where two threads could both
     * observe "absent" and both proceed to increment counters.
     */
    @Test
    void putIfAbsentAtomicUnderConcurrency() throws Exception {
        // decode step 10000 × sleep_scale 0.1 = 1000 ms, ensuring the first
        // scheduled completion is still in runningTasks when the remaining
        // threads arrive.
        MockPerformanceModel model = model("10", 10000.0);
        startCluster(model, 1, 1);

        JavaMockEngineCluster.FastRpcService decode = decodeServices.get(0);
        long requestId = 99L;
        int inputLen = 10;
        // Block-pool caliber: inputLen=10 rounds up to 1 block = 1024 tokens.
        long expectedKvTokens = 1024L;
        MockPerformanceModel.RequestShape shape = shapeOf(model, requestId, inputLen);

        int nThreads = 50;
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch allDone = new CountDownLatch(nThreads);
        AtomicInteger errors = new AtomicInteger(0);

        // All threads attempt to schedule the SAME requestId concurrently.
        for (int i = 0; i < nThreads; i++) {
            workerPool.submit(() -> {
                try {
                    startGate.await();
                    invokeScheduleDecodeCompletion(decode, shape, -1, null);
                } catch (Throwable t) {
                    errors.incrementAndGet();
                } finally {
                    allDone.countDown();
                }
            });
        }

        // Release all threads simultaneously to maximise scheduling overlap.
        startGate.countDown();
        assertTrue(allDone.await(10, TimeUnit.SECONDS),
                "all threads should complete within 10s, remaining: " + allDone.getCount());
        assertEquals(0, errors.get(), "no errors expected from concurrent scheduling");

        // Only the first thread should have incremented the counters.
        assertEquals(1, getActiveDecodeRequests(decode),
                "activeDecodeRequests should be 1, not " + nThreads
                        + " (putIfAbsent must reject duplicates)");
        assertEquals(expectedKvTokens, decode.getActiveKvTokens(),
                "activeKvTokens should be " + expectedKvTokens + ", not " + (nThreads * expectedKvTokens)
                        + " (putIfAbsent must reject duplicates)");
        assertEquals(1, decode.getInflightCount(),
                "pendingRequests should be 1, not " + nThreads
                        + " (putIfAbsent must reject duplicates)");
        assertEquals(1, decode.getRunningCount(),
                "runningTasks should contain 1 entry, not " + nThreads);

        // Wait for the single scheduled completion to drain all counters.
        awaitAllInflightZero(5_000);

        assertEquals(0, getActiveDecodeRequests(decode),
                "activeDecodeRequests should be 0 after completion");
        assertEquals(0, decode.getActiveKvTokens(),
                "activeKvTokens should be 0 after completion");
        assertEquals(0, decode.getInflightCount(),
                "pendingRequests should be 0 after completion");
        assertEquals(0, decode.getRunningCount(),
                "runningTasks should be empty after completion");
        assertFalse(decode.isLeakDetected(),
                "no leak should be detected after concurrent scheduling");
    }

    // ──────────── Cluster setup ────────────

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        for (int i = 0; i < nPrefill; i++) {
            int port = BASE_PORT + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            prefillServices.add(service);
        }

        for (int i = 0; i < nDecode; i++) {
            int port = BASE_PORT + nPrefill + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            decodeServices.add(service);
        }

        controlServer = new MockControlServer(
                services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
        controlServer.start();
    }

    // ──────────── Polling helpers ────────────

    private void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (services.values().stream()
                    .allMatch(s -> s.getInflightCount() == 0)) {
                return;
            }
            Thread.sleep(10);
        }
        StringBuilder sb = new StringBuilder("inflight not zero: ");
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            sb.append("port=").append(service.getGrpcPort())
                    .append(" inflight=").append(service.getInflightCount())
                    .append(" running=").append(service.getRunningCount())
                    .append(" ");
        }
        fail(sb.toString());
    }

    // ──────────── Reflection helpers ────────────

    /**
     * Reads the private {@code activeDecodeRequests} AtomicInteger field, which
     * has no public getter. This counter is the core of CodeReview fix #1.
     */
    private static int getActiveDecodeRequests(JavaMockEngineCluster.FastRpcService service)
            throws Exception {
        return MockEngineTestSupport.activeDecodeRequests(service);
    }

    /**
     * Invokes the private {@code scheduleDecodeCompletion} method via reflection.
     * This allows precise control over scheduling without going through the gRPC
     * enqueue/generateStream paths.
     */
    private static void invokeScheduleDecodeCompletion(
            JavaMockEngineCluster.FastRpcService service,
            MockPerformanceModel.RequestShape shape,
            long batchId,
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue)
            throws Exception {
        MockEngineTestSupport.scheduleDecodeCompletion(service, shape, batchId, responseQueue);
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String prefillFormula, double decodeStepMs)
            throws Exception {
        return MockEngineTestSupport.performanceModel(
                tempDir, prefillFormula, 0.1, decodeStepMs);
    }

    // ──────────── Shape helper ────────────

    /**
     * Builds a {@link MockPerformanceModel.RequestShape} for the given requestId
     * and input length, using a fresh empty cache (no prefix hits).
     */
    private static MockPerformanceModel.RequestShape shapeOf(
            MockPerformanceModel model, long requestId, int inputTokens) {
        return MockEngineTestSupport.requestShape(model, requestId, inputTokens);
    }

}
