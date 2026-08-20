package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Matrix sweep test for the Java mock engine.
 *
 * <p>Tests 5 cluster configurations (1P/1D, 1P/2D, 2P/2D, 2P/4D, 1P/4D)
 * with 3 concurrency levels (10, 50, 100) = 15 scenarios total.
 * Each scenario verifies: zero errors, all requests completed, no inflight leak.
 * Records TTFT p50/p99 and schedule latency, outputs a summary table.
 */
class MatrixSweepTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 62100;

    private static final int[][] CONFIGS = {
            {1, 1}, {1, 2}, {2, 2}, {2, 4}, {1, 4}
    };
    private static final int[] CONCURRENCIES = {10, 50, 100};

    @TempDir
    Path tempDir;

    @Test
    void matrixSweep() throws Exception {
        MockPerformanceModel model = model("10");
        List<ScenarioResult> results = new ArrayList<>();
        int portCounter = 0;

        System.out.println();
        System.out.println("=== FlexLB Mock Engine Matrix Sweep Test ===");
        System.out.println("Configurations: 1P/1D, 1P/2D, 2P/2D, 2P/4D, 1P/4D");
        System.out.println("Concurrency levels: 10, 50, 100");
        System.out.println("Total scenarios: " + (CONFIGS.length * CONCURRENCIES.length));
        System.out.println();

        for (int[] config : CONFIGS) {
            int nPrefill = config[0];
            int nDecode = config[1];
            for (int concurrency : CONCURRENCIES) {
                int basePort = BASE_PORT + portCounter;
                portCounter += nPrefill + nDecode + 5;

                ScenarioResult result = runScenario(model, nPrefill, nDecode, concurrency, basePort);
                results.add(result);

                String statusStr = result.passed() ? "PASS" : "FAIL";
                System.out.printf("  [%s] %dP/%dD concurrency=%d — errors=%d completed=%d/%d leak=%s"
                                + " ttft_p50=%dms ttft_p99=%dms schedule=%dms%n",
                        statusStr, nPrefill, nDecode, concurrency,
                        result.errorCount(), result.completedCount(), concurrency,
                        result.inflightLeak() ? "YES" : "NO",
                        result.ttftP50Ms(), result.ttftP99Ms(), result.scheduleLatencyMs());
            }
        }

        printSummaryTable(results);

        List<ScenarioResult> failures = results.stream()
                .filter(r -> !r.passed())
                .toList();
        if (!failures.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append(failures.size()).append(" scenario(s) failed:\n");
            for (ScenarioResult f : failures) {
                sb.append("  ").append(f.nPrefill()).append("P/").append(f.nDecode())
                        .append("D concurrency=").append(f.concurrency())
                        .append(": ").append(f.failureReason()).append("\n");
            }
            fail(sb.toString());
        }

        System.out.println("\nAll " + results.size() + " scenarios passed.");
    }

    // ──────────── Scenario runner ────────────

    private ScenarioResult runScenario(MockPerformanceModel model, int nPrefill, int nDecode,
                                       int concurrency, int basePort) throws Exception {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(8);
        MockControlServer controlServer = null;
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();

        try {
            List<JavaMockEngineCluster.FastRpcService> prefillServices = new ArrayList<>();
            List<JavaMockEngineCluster.FastRpcService> decodeServices = new ArrayList<>();

            for (int i = 0; i < nPrefill; i++) {
                int port = basePort + i;
                JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                        "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                        port, services, scheduler, model, 100,
                        new JavaMockEngineCluster.ClusterStats());
                services.put(port, service);
                prefillServices.add(service);
            }

            for (int i = 0; i < nDecode; i++) {
                int port = basePort + nPrefill + i;
                JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                        "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                        port, services, scheduler, model, 100,
                        new JavaMockEngineCluster.ClusterStats());
                services.put(port, service);
                decodeServices.add(service);
            }

            controlServer = new MockControlServer(services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
            controlServer.start();

            // ── Send requests ──
            long enqueueStartMs = System.currentTimeMillis();
            long scheduleStartNs = System.nanoTime();
            int totalErrors = 0;

            int requestsPerPrefill = concurrency / nPrefill;
            int remainder = concurrency % nPrefill;
            int requestIdCounter = 0;

            for (int i = 0; i < nPrefill; i++) {
                int count = requestsPerPrefill + (i < remainder ? 1 : 0);
                int startRequestId = requestIdCounter + 1;
                requestIdCounter += count;

                EngineRpcService.GenerateInputPB[] inputs =
                        new EngineRpcService.GenerateInputPB[count];
                for (int j = 0; j < count; j++) {
                    int decodePort = decodeServices.get(
                            (i * count + j) % nDecode).getGrpcPort();
                    inputs[j] = inputWithDecode(startRequestId + j, 10, decodePort);
                }
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefillServices.get(i), batch(1000 + i, slot(0, inputs)));
                totalErrors += response.getErrorsCount();
            }

            long scheduleEndNs = System.nanoTime();
            long scheduleLatencyMs = TimeUnit.NANOSECONDS.toMillis(scheduleEndNs - scheduleStartNs);

            // ── Wait for all completions ──
            awaitTotalCompleted(services, concurrency, 10_000);

            // ── Verify: no inflight leak ──
            boolean inflightLeak = false;
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                if (service.getInflightCount() != 0) {
                    inflightLeak = true;
                }
                if (service.isLeakDetected()) {
                    inflightLeak = true;
                }
            }

            // ── Collect TTFT from decode engines' finished tasks ──
            List<Long> ttfts = new ArrayList<>();
            for (JavaMockEngineCluster.FastRpcService decode : decodeServices) {
                EngineRpcService.WorkerStatusPB workerStatus = status(decode);
                for (EngineRpcService.TaskInfoPB task : workerStatus.getFinishedTaskListList()) {
                    ttfts.add(task.getEndTimeMs() - enqueueStartMs);
                }
            }

            Collections.sort(ttfts);
            long ttftP50 = ttfts.isEmpty() ? -1 : ttfts.get(ttfts.size() / 2);
            long ttftP99 = ttfts.isEmpty() ? -1
                    : ttfts.get((int) Math.min(
                            Math.ceil(ttfts.size() * 0.99) - 1, ttfts.size() - 1));

            // ── Determine pass/fail ──
            int completedCount = (int) services.values().stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();

            boolean passed = true;
            String failureReason = null;
            if (totalErrors != 0) {
                passed = false;
                failureReason = "error_count=" + totalErrors + " (expected 0)";
            }
            if (completedCount != concurrency) {
                passed = false;
                failureReason = (failureReason == null ? "" : failureReason + "; ")
                        + "completed=" + completedCount + " (expected " + concurrency + ")";
            }
            if (inflightLeak) {
                passed = false;
                failureReason = (failureReason == null ? "" : failureReason + "; ")
                        + "inflight leak detected";
            }

            return new ScenarioResult(nPrefill, nDecode, concurrency,
                    totalErrors, completedCount, inflightLeak,
                    ttftP50, ttftP99, scheduleLatencyMs,
                    System.currentTimeMillis() - enqueueStartMs,
                    passed, failureReason);
        } finally {
            if (controlServer != null) {
                controlServer.stop();
            }
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    // ──────────── Summary table ────────────

    private void printSummaryTable(List<ScenarioResult> results) {
        System.out.println();
        System.out.println("+---------+---------+------------+--------+------------+--------------+"
                + "----------+----------+--------+");
        System.out.println("| PREFILL | DECODE  | CONCURRENCY| ERRORS | COMPLETED  | INFLIGHT LEAK|"
                + " TTFT P50 | TTFT P99 | STATUS |");
        System.out.println("+---------+---------+------------+--------+------------+--------------+"
                + "----------+----------+--------+");
        for (ScenarioResult r : results) {
            System.out.printf("|   %d     |   %d     |    %3d     |  %4d   |   %3d/%-3d  |    %5s     |"
                            + "  %5d ms|  %5d ms|  %s  |%n",
                    r.nPrefill(), r.nDecode(), r.concurrency(),
                    r.errorCount(), r.completedCount(), r.concurrency(),
                    r.inflightLeak() ? "YES" : "NO",
                    r.ttftP50Ms(), r.ttftP99Ms(),
                    r.passed() ? "PASS" : "FAIL");
        }
        System.out.println("+---------+---------+------------+--------+------------+--------------+"
                + "----------+----------+--------+");

        long passed = results.stream().filter(ScenarioResult::passed).count();
        long failed = results.size() - passed;
        System.out.printf("%nTotal: %d scenarios — %d passed, %d failed%n",
                results.size(), passed, failed);
    }

    // ──────────── Polling helpers ────────────

    private void awaitTotalCompleted(
            Map<Integer, JavaMockEngineCluster.FastRpcService> services,
            int expected, long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            long completed = services.values().stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            if (completed >= expected) {
                return;
            }
            Thread.sleep(10);
        }
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, formula);
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    // ──────────── Protobuf builders ────────────

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

    private static EngineRpcService.WorkerStatusPB status(
            JavaMockEngineCluster.FastRpcService service) {
        return unary(observer -> service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(0)
                        .build(),
                observer));
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

    // ──────────── Result record ────────────

    private record ScenarioResult(int nPrefill, int nDecode, int concurrency,
                                   int errorCount, int completedCount, boolean inflightLeak,
                                   long ttftP50Ms, long ttftP99Ms, long scheduleLatencyMs,
                                   long totalDurationMs, boolean passed, String failureReason) {
    }
}
