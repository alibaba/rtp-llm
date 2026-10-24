package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/** Direct prefill requests use the real FIFO scheduler budgets and resource boundaries. */
class DirectPrefillCoalescingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63200;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private JavaMockEngineCluster.ClusterStats stats;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "direct-coalesce-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        services = new ConcurrentHashMap<>();
        stats = new JavaMockEngineCluster.ClusterStats();
    }

    @AfterEach
    void tearDown() throws InterruptedException {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            service.shutdown();
        }
        scheduler.shutdownNow();
        scheduler.awaitTermination(3, TimeUnit.SECONDS);
    }

    // ──────────── Test 1: queued direct requests coalesce into one batch ────────────

    @Test
    void queuedDirectRequestsCoalesceIntoOneBatch() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill =
                startPrefill(model(4));

        // Fire all 6 generate_stream calls up front (admission is synchronous,
        // responses arrive on the response executor): #1 takes the only slot,
        // #2-#6 park in the direct queue within #1's 100 ms prefill window.
        // A serial await-per-request loop would never leave anything queued.
        List<CompletableFuture<Throwable>> futures = new ArrayList<>();
        for (int i = 1; i <= 6; i++) {
            futures.add(generateAsync(prefill, input(1000 + i, 10)));
        }
        for (int i = 0; i < futures.size(); i++) {
            assertNull(futures.get(i).get(10, TimeUnit.SECONDS),
                    "request " + (i + 1) + " should be admitted");
        }

        awaitCompleted(prefill, 6, 5_000);
        assertEquals(6, prefill.getCompletedCount());
        assertEquals(0, prefill.getWaitingCount());
        assertEquals(0, prefill.getInflightCount());

        // Batch structure: [1] + [2,3,4,5] + [6] — coalescing is observable in
        // the per-engine prefill batch stats (java_mock_stats batch counters).
        assertEquals(3, stats.prefillBatches.sum(),
                "expected 1 + 4 + 1 batches after coalescing");
        assertEquals(6, stats.prefillBatchRequests.sum());
        assertEquals(4, stats.maxPrefillBatchSize.get(),
                "the coalesced batch should hold 4 requests (max_generate_batch_size)");
        @SuppressWarnings("unchecked")
        List<Long> buckets = (List<Long>) prefill.getSnapshot().get("prefill_batch_size_buckets");
        assertEquals(2L, buckets.get(0)); // two single-request batches
        assertEquals(2L, buckets.get(1)); // no batch of size 2
        assertEquals(3L, buckets.get(2)); // the remaining batch has size 4
        assertEquals(3L, buckets.get(buckets.size() - 1)); // +Inf equals count
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Test 2: max_generate_batch_size=1 restores one-request batches ────────────

    @Test
    void maxOfOneRestoresSingleRequestBatches() throws Exception {
        JavaMockEngineCluster.FastRpcService prefill =
                startPrefill(model(1));

        List<CompletableFuture<Throwable>> futures = new ArrayList<>();
        for (int i = 1; i <= 5; i++) {
            futures.add(generateAsync(prefill, input(2000 + i, 10)));
        }
        for (CompletableFuture<Throwable> future : futures) {
            assertNull(future.get(10, TimeUnit.SECONDS));
        }

        awaitCompleted(prefill, 5, 5_000);
        assertEquals(5, prefill.getCompletedCount());
        assertEquals(5, stats.prefillBatches.sum(),
                "max_generate_batch_size=1 limits each forward to one stream");
        assertEquals(5, stats.prefillBatchRequests.sum());
        assertEquals(1, stats.maxPrefillBatchSize.get());
        assertFalse(prefill.isLeakDetected());
    }

    // ──────────── Harness ────────────

    @Test void fifoDefersPhysicalKvUntilBatchSelection() throws Exception {
        verifyFifoPhysicalPool(false);
    }

    @Test void fifoCancelsWaitingRequestWithoutAllocatingKv() throws Exception {
        verifyFifoPhysicalPool(true);
    }

    @Test void fifoRespectsInitializedStreamLimitWithSpareBlocks() throws Exception {
        verifyFifoPhysicalPool(false, 100, 1);
    }

    @Test void fifoRunningCancelKeepsKvUntilForwardBoundary() throws Exception {
        var model = model(1);
        model.setOverrideFixedPrefillMs(500.0);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
        var prefill = startPrefill(model);
        var first = generateAsync(prefill, input(7101, 10));
        var second = generateAsync(prefill, input(7102, 10));
        long allocated = prefill.getActiveKvTokens();
        assertTrue(allocated > 0);
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, prefill.cancel(7101));
        assertEquals(allocated, prefill.getActiveKvTokens(), "in-flight forward still owns its KV");
        assertEquals(2, prefill.getInflightCount(), "terminal is committed at the execution boundary");
        assertNotNull(first.get(5, TimeUnit.SECONDS));
        assertNull(second.get(5, TimeUnit.SECONDS));
        awaitCompleted(prefill, 1, 5000);
        assertEquals(0, prefill.getActiveKvTokens());
    }

    @Test void consumerSeesCancelBeforeSchedulerCommitAndLaterDeadlineCannotReplaceIt() throws Exception {
        var model = model(1);
        model.setOverrideFixedPrefillMs(400.0);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
        var service = startPrefill(model);
        var request = input(7301, 10).toBuilder().setGenerateConfig(input(7301, 10).getGenerateConfig()
                .toBuilder().setTimeoutMs(100)).build();
        var response = generateAsync(service, request);
        long held = service.getActiveKvTokens();
        service.cancel(7301);
        var error = response.get(200, TimeUnit.MILLISECONDS);
        assertEquals(io.grpc.Status.Code.CANCELLED, io.grpc.Status.fromThrowable(error).getCode());
        assertEquals(8100, MockRpcErrors.downstream(error).getErrorCode());
        assertEquals(held, service.getActiveKvTokens());
        assertEquals(1, service.getInflightCount());
        assertEquals(0, service.getCancelledCount());
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (service.getInflightCount() != 0 && System.nanoTime() < until) Thread.sleep(2);
        var status = MockEngineTestSupport.workerStatus(service, 0);
        assertEquals(8100, status.getFinishedTaskList(0).getErrorInfo().getErrorCode());
        assertEquals(0, service.getActiveKvTokens());
    }

    @Test void permanentInitFailureTerminatesInSchedulerWithoutCountingAsCancel() throws Exception {
        var model = model(1);
        model.observedReserveBlocks = 1;
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
        var service = new JavaMockEngineCluster.FastRpcService("prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                BASE_PORT, services, scheduler, model, 5, stats);
        services.put(BASE_PORT, service);
        var failure = generateAsync(service, input(7351, 5120)).get(2, TimeUnit.SECONDS);
        assertEquals(602, MockRpcErrors.downstream(failure).getErrorCode());
        assertEquals(0, service.getInflightCount());
        assertEquals(0, service.getCancelledCount());
        assertEquals(0, service.getActiveKvTokens());
        assertEquals(602, MockEngineTestSupport.workerStatus(service, 0).getFinishedTaskList(0).getErrorInfo().getErrorCode());
    }

    @Test void fifoBatchRpcPreservesGroupsAcrossExecutionBoundaries() throws Exception {
        var model = model(1);
        model.setOverrideFixedPrefillMs(200.0);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
        var prefill = startPrefill(model);
        var blocker = generateAsync(prefill, input(7201, 10));
        for (int batch = 0; batch < 2; batch++) {
            var response = MockEngineTestSupport.enqueueAndFetch(prefill, MockEngineTestSupport.batch(100 + batch,
                    MockEngineTestSupport.slot(0, input(7210 + batch * 2, 10), input(7211 + batch * 2, 10))));
            assertEquals(2, response.getSuccessesCount());
        }
        assertNull(blocker.get(5, TimeUnit.SECONDS));
        awaitCompleted(prefill, 5, 5000);
        assertEquals(3, stats.prefillBatches.sum(), "one blocker plus two isolated groups; no cross-group regroup");
        assertEquals(2, stats.maxPrefillBatchSize.get());
        var status = MockEngineTestSupport.workerStatus(prefill, 0);
        assertEquals(0, status.getAvailableConcurrency(), "real LocalRpcServer leaves legacy count fields unset");
        assertEquals(0, status.getWaitingQueryLen());
        assertEquals(0, status.getRunningQueryLen());
        assertFalse(prefill.isLeakDetected());
    }

    private void verifyFifoPhysicalPool(boolean cancelLast) throws Exception {
        verifyFifoPhysicalPool(cancelLast, 5, 128);
    }

    private void verifyFifoPhysicalPool(boolean cancelLast, int blocks, int initializedCap) throws Exception {
        Path config = tempDir.resolve("pool-fifo.json");
        Path master = tempDir.resolve("pool-master.json");
        MAPPER.writeValue(config.toFile(), Map.of("block_size", 1024,
                "prefill", Map.of("fifo", Map.of("max_generate_batch_size", 64,
                        "max_batch_tokens_size", 100000,
                        "max_seq_len", 10000,
                        "max_inited_kv_cache_streams", initializedCap)),
                "decode", Map.of("step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, "500");
        var model = MockModelFixture.load(config.toString(), master.toString());
        var prefill = new JavaMockEngineCluster.FastRpcService("prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL, BASE_PORT, services,
                scheduler, model, blocks, stats);
        services.put(BASE_PORT, prefill);
        var first = generateAsync(prefill, input(5001, 3500));
        var second = generateAsync(prefill, input(5002, 3500));
        var last = generateAsync(prefill, input(5003, 3500));
        assertFalse(second.isDone(), "temporary P KV shortage must queue, not reject");
        assertFalse(last.isDone());
        assertEquals(2, prefill.getWaitingCount());
        if (cancelLast) {
            prefill.cancel(5003);
            assertEquals(3, prefill.getInflightCount(), "consumer error must not commit scheduler FINISHED early");
            assertEquals(0, prefill.getCancelledCount());
        }
        assertNull(first.get(5, TimeUnit.SECONDS));
        assertNull(second.get(5, TimeUnit.SECONDS));
        if (cancelLast) assertNotNull(last.get(5, TimeUnit.SECONDS));
        else assertNull(last.get(5, TimeUnit.SECONDS));
        awaitCompleted(prefill, cancelLast ? 2 : 3, 5000);
        assertEquals(1, stats.maxPrefillBatchSize.get(), "KV capacity or initialized-stream cap limits batch to one");
        assertEquals(0, prefill.getWaitingCount());
        assertEquals(blocks * 1024L, ((Number) prefill.getSnapshot().get("available_kv_tokens")).longValue());
        assertFalse(prefill.isLeakDetected());
    }

    @Test
    void productionFifoCoalescesNonBatchUnderCandidateBudget() throws Exception {
        Path performance = tempDir.resolve("fifo.json");
        Path master = tempDir.resolve("fifo-master.json");
        MAPPER.writeValue(performance.toFile(), Map.of("block_size", 1024,
                "prefill", Map.of("fifo", Map.of("max_generate_batch_size", 64,
                        "max_batch_tokens_size", 100,
                        "max_seq_len", 10000)),
                "decode", Map.of("step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, "500");
        var loaded = MockModelFixture.load(performance.toString(), master.toString());
        assertNotNull(loaded.forEngine().prefillBatchPolicy(), "engine clone preserves policy");
        var prefill = startPrefill(loaded.forEngine());
        var first = generateAsync(prefill, input(4000, 1));
        var a = generateAsync(prefill, input(4001, 40));
        var skipped = generateAsync(prefill, input(4002, 60));
        var smaller = generateAsync(prefill, input(4003, 39));
        assertNull(first.get(5, TimeUnit.SECONDS));
        assertNull(a.get(5, TimeUnit.SECONDS));
        assertNull(smaller.get(5, TimeUnit.SECONDS));
        assertFalse(skipped.isDone(), "60-token candidate exceeds sum and rectangle; later 39 fits");
        assertNull(skipped.get(5, TimeUnit.SECONDS));
        awaitCompleted(prefill, 4, 5000);
        assertEquals(3, stats.prefillBatches.sum());
        assertEquals(2, stats.maxPrefillBatchSize.get());
        assertEquals(0, prefill.getWaitingCount());
        assertFalse(prefill.isLeakDetected());
    }

    @Test void fifoRematchesCachePopulatedWhileWaiting() throws Exception {
        Path config = tempDir.resolve("cache-fifo.json");
        Path master = tempDir.resolve("cache-master.json");
        MAPPER.writeValue(config.toFile(), Map.of("block_size", 16,
                "prefill", Map.of("fifo", Map.of("max_generate_batch_size", 64,
                        "max_batch_tokens_size", 100,
                        "max_seq_len", 10000)),
                "decode", Map.of("step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, "500");
        var model = MockModelFixture.load(config.toString(), master.toString());
        model.nativeTokenCacheKeys = true;
        var prefill = startPrefill(model);
        var first = generateAsync(prefill, input(6001, 60));
        var second = generateAsync(prefill, input(6002, 60));
        var third = generateAsync(prefill, input(6003, 60));
        assertNull(first.get(5, TimeUnit.SECONDS));
        assertNull(second.get(5, TimeUnit.SECONDS));
        assertNull(third.get(5, TimeUnit.SECONDS));
        awaitCompleted(prefill, 3, 5000);
        assertEquals(3, stats.prefillBatches.sum(),
                "cache reduces compute but full 60+60 and rectangle still exceed the 100-token limit");
        assertEquals(1, stats.maxPrefillBatchSize.get());
        assertEquals(96L, ((Number) prefill.getSnapshot().get("hit_tokens_total")).longValue(),
                "waiting requests still rematch the 48-token prefix at execution");
        assertFalse(prefill.isLeakDetected());
    }

    private JavaMockEngineCluster.FastRpcService startPrefill(MockPerformanceModel model) {
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                BASE_PORT, services, scheduler, model, 100, stats);
        services.put(BASE_PORT, service);
        return service;
    }

    private MockPerformanceModel model(int maxRequests)
            throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of(
                        "scale", 1.0,
                        "fifo", Map.of("max_generate_batch_size", maxRequests, "max_batch_tokens_size", 1_048_576, "max_seq_len", 1_048_576)),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, "100");
        return MockModelFixture.load(performance.toString(), master.toString());
    }

    private static void awaitCompleted(JavaMockEngineCluster.FastRpcService service,
                                       int expected, long timeoutMs)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getCompletedCount() >= expected && service.getInflightCount() == 0) {
                return;
            }
            Thread.sleep(10);
        }
        fail("not drained: completed=" + service.getCompletedCount()
                + " inflight=" + service.getInflightCount()
                + " waiting=" + service.getWaitingCount());
    }

    private static EngineRpcService.GenerateInputPB input(long requestId, int inputTokens) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setReuseCache(true).setEnableDeviceCache(true)
                        .setMaxNewTokens(1)
                        .build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    /**
     * Fires generate_stream and returns a future completed with the stream
     * error (or null on success). Admission runs synchronously on the caller
     * thread; the single-frame response / error arrives on the response
     * executor, so callers can fire many requests before awaiting any.
     */
    private static CompletableFuture<Throwable> generateAsync(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.GenerateInputPB input) {
        CompletableFuture<Throwable> future = new CompletableFuture<>();
        service.generateStreamCall(input, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
                if (value.hasErrorInfo() && value.getErrorInfo().getErrorCodeValue() != 0) {
                    future.complete(new IllegalStateException(value.getErrorInfo().toString()));
                }
            }

            @Override
            public void onError(Throwable throwable) {
                future.complete(throwable);
            }

            @Override
            public void onCompleted() {
                future.complete(null);
            }
        });
        return future;
    }
}
