package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.lang.reflect.Field;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Leak-attribution regression tests for the typed CANCELLED terminal frame.
 *
 * <p>The cancel() main path used to publish its terminal without the
 * EnqueueBatch identity, so the master's reconcile could not correlate the
 * cancel with the batch it displaced and dropped the terminal into the
 * batch-mismatch dead branch — leaking the inflight slot (the new-base
 * counterpart of the legacy finishedByBatch[0] leak). The fix mirrors the
 * upstream positiveLifecycleBatchId pattern already used by
 * recordPriorityPreemptionCanceled.</p>
 *
 * <p>Also pins the A1/A2 cancel census counters exported through
 * java_mock_stats: tracked / already-finished / unknown branch distribution.</p>
 */
class CancelTerminalBatchIdTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 63300;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private JavaMockEngineCluster.ClusterStats stats;
    private JavaMockEngineCluster.FastRpcService prefill;

    @BeforeEach
    void setUp() {
        scheduler = Executors.newScheduledThreadPool(4, runnable -> {
            Thread thread = new Thread(runnable, "cancel-batch-id-scheduler");
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

    @Test
    void cancelTerminalCarriesEnqueueBatchIdentity() throws Exception {
        // Prefill expression 20000ms * sleep_scale 0.1 = 2000ms — a wide
        // execution window so cancel() reliably lands while the batch runs.
        MockPerformanceModel model = model("20000");
        newPrefillService(model);

        long requestId = 100L;
        long batchId = 77L;
        enqueue(prefill, batch(batchId, slot(0, input(requestId, 16))));

        // Cancel must land while the request is still tracked.
        EngineRpcService.TaskPhase phase = prefill.cancel(requestId);
        assertNotNull(phase, "cancel should observe a tracked prefill request");

        EngineRpcService.TaskInfoPB terminal = completionOf(requestId);
        assertNotNull(terminal, "a terminal for the cancelled request must be published");
        assertEquals(EngineRpcService.ErrorCodePB.CANCELLED.getNumber(),
                terminal.getErrorInfo().getErrorCode(),
                "terminal must carry the CANCELLED error code");
        assertEquals(batchId, terminal.getBatchId(),
                "typed CANCELLED terminal must carry the exact EnqueueBatch "
                        + "identity (leak fix: master reconcile correlation)");
    }

    @Test
    void cancelRequestCensusCountsAllBranches() throws Exception {
        // Prefill expression 20000ms * sleep_scale 0.1 = 2000ms — a wide
        // execution window so cancelRequest reliably lands while tracked.
        MockPerformanceModel slowModel = model("20000");
        newPrefillService(slowModel);

        // Unknown request: cancel for an id the engine never saw.
        prefill.cancelRequest(9_999L);
        assertEquals(1L, stats.cancelCensusUnknown.sum(),
                "unknown-branch cancel must be counted");

        // Tracked request: cancelRequest while the batch is running. The
        // tracked branch routes through the priority-preemption cancel path,
        // which also arms the tombstone for later re-cancels.
        long requestId = 200L;
        enqueue(prefill, batch(88L, slot(0, input(requestId, 16))));
        JavaMockEngineCluster.CancelResult tracked = prefill.cancelRequest(requestId);
        assertTrue(tracked.found(), "cancelRequest must find the tracked request");
        assertEquals(1L, stats.cancelCensusTracked.sum(),
                "tracked-branch cancel must be counted");

        // Re-cancel of the same id hits the priority-cancel tombstone armed by
        // the tracked branch (the engine keeps its one-shot cancel semantics).
        JavaMockEngineCluster.CancelResult tombstone = prefill.cancelRequest(requestId);
        assertTrue(tombstone.found() && tombstone.alreadyFinished(),
                "tombstone re-cancel must report found + already-finished");
        assertEquals(1L, stats.cancelCensusTombstone.sum(),
                "tombstone-branch re-cancel must be counted");

        // And the typed terminal from the tracked branch carries the batch id.
        EngineRpcService.TaskInfoPB terminal = completionOf(requestId);
        assertNotNull(terminal);
        assertEquals(88L, terminal.getBatchId(),
                "tracked-branch CANCELLED terminal must carry the batch identity");

        // Already-finished: a request that ran to its NATURAL completion, then
        // received a late cancel (no tombstone was ever armed).
        MockPerformanceModel fastModel = model("10");
        newPrefillService(fastModel);
        long finishedId = 300L;
        enqueue(prefill, batch(99L, slot(0, input(finishedId, 16))));
        awaitTerminal(finishedId);
        JavaMockEngineCluster.CancelResult finished = prefill.cancelRequest(finishedId);
        assertTrue(finished.alreadyFinished(),
                "late cancel must report the already-finished branch");
        assertEquals(1L, stats.cancelCensusAlreadyFinished.sum(),
                "already-finished-branch cancel must be counted");
    }

    private void awaitTerminal(long requestId) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(10_000);
        while (System.nanoTime() < deadline) {
            if (completionOf(requestId) != null) {
                return;
            }
            Thread.sleep(10);
        }
    }

    // ──────────── Service / model helpers ────────────

    private void newPrefillService(MockPerformanceModel model) {
        int port = BASE_PORT + services.size();
        prefill = new JavaMockEngineCluster.FastRpcService(
                "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                port, services, scheduler, model, 100, stats);
        services.put(port, prefill);
    }

    private MockPerformanceModel model(String prefillExpression) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 0.1,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                // Explicit decode timing (1 ms/token): decode latency is
                // irrelevant to this cancel-census test, but the model now
                // rejects a silent default decode curve.
                "decode", Map.of("scale", 1.0, "tokens_per_step", 1.0,
                        "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MockMasterConfig.writeWithPrefillExpression(master, prefillExpression);
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    // ──────────── Terminal-frame access (reflection on completions queue) ────────────

    private EngineRpcService.TaskInfoPB completionOf(long requestId) throws Exception {
        Field field = JavaMockEngineCluster.FastRpcService.class
                .getDeclaredField("completions");
        field.setAccessible(true);
        ConcurrentLinkedQueue<?> queue =
                (ConcurrentLinkedQueue<?>) field.get(prefill);
        Object latest = null;
        for (Object element : queue) {
            Field taskField = element.getClass().getDeclaredField("task");
            taskField.setAccessible(true);
            EngineRpcService.TaskInfoPB task =
                    (EngineRpcService.TaskInfoPB) taskField.get(element);
            if (task.getRequestId() == requestId) {
                latest = task;
            }
        }
        return (EngineRpcService.TaskInfoPB) latest;
    }

    // ──────────── Protobuf / RPC helpers ────────────

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

    private static void enqueue(JavaMockEngineCluster.FastRpcService service,
                                EngineRpcService.EnqueueBatchRequestPB request) {
        service.enqueueBatch(request, new io.grpc.stub.StreamObserver<EngineRpcService.EnqueueBatchResponsePB>() {
            @Override
            public void onNext(EngineRpcService.EnqueueBatchResponsePB value) {
            }

            @Override
            public void onError(Throwable t) {
            }

            @Override
            public void onCompleted() {
            }
        });
    }
}
