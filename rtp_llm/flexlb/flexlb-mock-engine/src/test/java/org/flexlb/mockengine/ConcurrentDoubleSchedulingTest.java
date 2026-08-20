package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
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
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Concurrent double-scheduling stress test for the Java mock engine.
 *
 * <p>Simulates the exact scenario discovered in Chris's load test: with a 2P/4D
 * cluster (2 prefill + 4 decode engines) and high replay speed, requests are
 * scheduled simultaneously via both {@code enqueueBatch} (prefill → decode)
 * and {@code generateStreamCall} (direct decode) paths for the same requestId.
 *
 * <p>The fix uses {@code putIfAbsent} in {@code scheduleDecodeCompletion} to
 * atomically check-and-insert, preventing double-counting of pendingRequests
 * and runningTasks. These tests verify the fix holds under high concurrency.
 *
 * <p>Test 1 ({@link #concurrentEnqueueAndGenerateStreamNoLeak}) exercises the
 * double-scheduling race: 100 requests, each scheduled via both paths
 * concurrently. The decode step is set high enough (100 ms with sleep_scale
 * 0.1) to ensure the first {@code scheduleDecodeCompletion} entry is still in
 * runningTasks when the second path arrives, maximising putIfAbsent collisions.
 *
 * <p>Test 2 ({@link #highConcurrencySinglePathNoLeak}) is a control group:
 * 100 requests all via {@code enqueueBatch} (single path), 100 concurrent
 * threads, verifying no leak without double-scheduling.
 */
class ConcurrentDoubleSchedulingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 62800;

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
        scheduler = Executors.newScheduledThreadPool(16, runnable -> {
            Thread thread = new Thread(runnable, "concurrent-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        workerPool = Executors.newFixedThreadPool(200, runnable -> {
            Thread thread = new Thread(runnable, "concurrent-test-worker");
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

    // ──────────── Test 1: Concurrent Double Scheduling — No Leak ────────────

    @Test
    void concurrentEnqueueAndGenerateStreamNoLeak() throws Exception {
        // Prefill 1 ms (formula "10" × sleep_scale 0.1) and decode 100 ms
        // (step 1000.0 × sleep_scale 0.1). With 50 requests per prefill engine
        // the total prefill time is 50 ms, well below the 100 ms decode window.
        // This guarantees that when Path A's prefill finishes and calls
        // startDecode → scheduleDecodeCompletion, Path B's decode entry is
        // still in runningTasks, so putIfAbsent correctly rejects the duplicate.
        MockPerformanceModel model = model("10", 1000.0);
        startCluster(model, 2, 4);

        int n = 100;
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch allDone = new CountDownLatch(n * 2);
        AtomicInteger errors = new AtomicInteger(0);

        for (int i = 0; i < n; i++) {
            final long requestId = i + 1;
            final int prefillIdx = i % prefillServices.size();
            final int decodeIdx = i % decodeServices.size();
            final JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(prefillIdx);
            final JavaMockEngineCluster.FastRpcService decode = decodeServices.get(decodeIdx);
            final int decodePort = decode.getGrpcPort();
            final long batchId = 30000L + i;

            // Path A: enqueueBatch → schedulePrefillCompletion → prefill done
            //         → startDecode → decode.scheduleDecodeCompletion
            workerPool.submit(() -> {
                try {
                    startGate.await();
                    EngineRpcService.GenerateInputPB input =
                            inputWithDecode(requestId, 10, decodePort);
                    enqueue(prefill, batch(batchId, slot(0, input)));
                } catch (Throwable t) {
                    errors.incrementAndGet();
                } finally {
                    allDone.countDown();
                }
            });

            // Path B: generateStreamCall on decode engine with the SAME requestId
            //         → decode.scheduleDecodeCompletion (immediate)
            //
            // Fire-and-forget: we don't wait for the stream response because
            // when Path A wins the putIfAbsent race, the response is offered
            // to the prefill engine's queue (passed via startDecode), not the
            // decode engine's queue that generateStreamCall's poller waits on.
            // The test's goal is to verify no inflight leak, not response
            // delivery. The poller threads are cleaned up in tearDown via
            // service.shutdown() → responseExecutor.shutdownNow().
            workerPool.submit(() -> {
                try {
                    startGate.await();
                    EngineRpcService.GenerateInputPB input = input(requestId, 10);
                    decode.generateStreamCall(input, new StreamObserver<>() {
                        @Override
                        public void onNext(EngineRpcService.GenerateOutputsPB value) { }

                        @Override
                        public void onError(Throwable throwable) { }

                        @Override
                        public void onCompleted() { }
                    });
                } catch (Throwable t) {
                    errors.incrementAndGet();
                } finally {
                    allDone.countDown();
                }
            });
        }

        // Release all 200 tasks simultaneously to maximise scheduling overlap
        startGate.countDown();

        // Wait for all tasks (100 enqueue + 100 generateStream) to return
        assertTrue(allDone.await(15, TimeUnit.SECONDS),
                "all tasks should complete within 15s, remaining: " + allDone.getCount());

        // Wait for all inflight to drain (prefill + decode completions)
        awaitAllInflightZero(5_000);

        // Verify pendingRequests == 0 on all engines
        assertAllInflightZero();

        // Verify runningTasks is empty on all engines
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertEquals(0, service.getRunningCount(),
                    "runningTasks should be empty for engine on port " + service.getGrpcPort());
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected on engine on port " + service.getGrpcPort());
        }

        assertEquals(0, errors.get(), "no errors should occur, got " + errors.get());
    }

    // ──────────── Test 2: High Concurrency Single Path — No Leak (Control) ────────────

    @Test
    void highConcurrencySinglePathNoLeak() throws Exception {
        // Single-path control: all requests via enqueueBatch only, no
        // double-scheduling. Verifies the cluster handles 100 concurrent
        // requests without inflight leak.
        MockPerformanceModel model = model("10", 1.0);
        startCluster(model, 2, 4);

        int n = 100;
        CountDownLatch startGate = new CountDownLatch(1);
        CountDownLatch allDone = new CountDownLatch(n);
        AtomicInteger errors = new AtomicInteger(0);

        for (int i = 0; i < n; i++) {
            final long requestId = i + 1;
            final int prefillIdx = i % prefillServices.size();
            final int decodeIdx = i % decodeServices.size();
            final JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(prefillIdx);
            final int decodePort = decodeServices.get(decodeIdx).getGrpcPort();
            final long batchId = 40000L + i;

            workerPool.submit(() -> {
                try {
                    startGate.await();
                    EngineRpcService.GenerateInputPB input =
                            inputWithDecode(requestId, 10, decodePort);
                    enqueue(prefill, batch(batchId, slot(0, input)));
                } catch (Throwable t) {
                    errors.incrementAndGet();
                } finally {
                    allDone.countDown();
                }
            });
        }

        startGate.countDown();

        assertTrue(allDone.await(10, TimeUnit.SECONDS),
                "all tasks should complete within 10s, remaining: " + allDone.getCount());

        awaitAllInflightZero(5_000);

        assertAllInflightZero();

        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertEquals(0, service.getRunningCount(),
                    "runningTasks should be empty for engine on port " + service.getGrpcPort());
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected on engine on port " + service.getGrpcPort());
        }

        assertEquals(0, errors.get(), "no errors should occur, got " + errors.get());
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

        controlServer = new MockControlServer(services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
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

    private void assertAllInflightZero() {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertEquals(0, service.getInflightCount(),
                    "inflight should be 0 for engine on port " + service.getGrpcPort());
        }
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String prefillFormula, double decodeStepMs) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 0.1,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, decodeStepMs)))));
        MockMasterConfig.writeWithPrefillExpression(master, prefillFormula);
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    // ──────────── Protobuf builders ────────────

    private static EngineRpcService.GenerateInputPB input(long requestId, int inputTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    private static EngineRpcService.GenerateInputPB inputWithDecode(
            long requestId, int inputTokens, int decodePort) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                                .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                .setRoleStr("DECODE")
                                .setGrpcPort(decodePort)
                                .build())
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    private static EngineRpcService.EnqueueBatchDpSlotPB slot(
            int dpRank, EngineRpcService.GenerateInputPB... inputs) {
        EngineRpcService.EnqueueBatchDpSlotPB.Builder slot =
                EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(dpRank);
        for (EngineRpcService.GenerateInputPB input : inputs) {
            slot.addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                    .setInput(input)
                    .build());
        }
        return slot.build();
    }

    private static EngineRpcService.EnqueueBatchRequestPB batch(
            long batchId, EngineRpcService.EnqueueBatchDpSlotPB... slots) {
        return EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                .setBatchId(batchId)
                .addAllDpSlots(List.of(slots))
                .build();
    }

    // ──────────── RPC helpers ────────────

    private static EngineRpcService.EnqueueBatchResponsePB enqueue(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.EnqueueBatchRequestPB request) {
        return unary(observer -> service.enqueueBatch(request, observer));
    }

    private static <T> T unary(Consumer<StreamObserver<T>> invocation) {
        AtomicReference<T> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        invocation.accept(new StreamObserver<>() {
            @Override
            public void onNext(T value) {
                response.set(value);
            }

            @Override
            public void onError(Throwable throwable) {
                error.set(throwable);
                latch.countDown();
            }

            @Override
            public void onCompleted() {
                latch.countDown();
            }
        });
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                fail("unary response timeout");
            }
        } catch (InterruptedException e) {
            fail("interrupted waiting for unary response");
        }
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        assertNotNull(response.get(), "unary response");
        return response.get();
    }
}
