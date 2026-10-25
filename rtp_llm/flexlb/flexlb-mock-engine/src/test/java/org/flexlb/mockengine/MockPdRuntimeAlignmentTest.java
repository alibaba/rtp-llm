package org.flexlb.mockengine;

import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** Real references: DecodeRpcServer::allocateResource/localGenerate,
 * PrefillRpcServer::remoteLoadCacheEnd/preferPriorityPreemption, RpcServerRuntimeMeta. */
class MockPdRuntimeAlignmentTest {
    @TempDir Path dir;

    @Test void prefillRuntimeLivesUntilFetchAndDecodeConsumesOnlyRemainingTokens() throws Exception {
        var model = performanceModel(dir, "10", 1, 300, Map.of(), Map.of("tokens_per_step", 1));
        model.nativeTokenCacheKeys = true;
        try (var c = MockEngineTestCluster.create(model, 63820, 1, 1)) {
            var p = c.prefill(0); var d = c.decode(0);
            p.setAutoFetch(false);
            assertEquals(1, enqueue(p, batch(77, slot(0, inputWithDecode(1, 1024, d.getGrpcPort(), 2))))
                    .getSuccessesCount());
            c.awaitNoInflight(p, 2000);
            var before = workerStatus(p, 0);
            assertEquals(0, before.getFinishedTaskListCount(), "compute completion is not runtime dequeue");
            assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_KV_ALLOCATED, before.getRunningTaskInfo(0).getPhase());
            assertEquals(0, d.getActiveDecodeCount());
            assertEquals(-1, workerStatus(d, 0).getRunningTaskInfo(0).getBatchId());
            var output = fetch(p, 1);
            await(() -> d.getActiveDecodeCount() == 1);
            synchronized (field(d, "decodeQueueLock")) {
                var stream = ((Map<?, ?>) field(d, "decodeRunning")).get(1L);
                assertEquals(1.0, field(stream, "generatedTokens"));
                assertEquals(1, field(stream, "totalSteps"));
                assertEquals(2048, d.getActiveKvTokens(), "reserve next forward KV before the forward completes");
            }
            output.done.get(3, TimeUnit.SECONDS);
            c.awaitAllInflightZero(2000);
            var pDone = workerStatus(p, 0).getFinishedTaskList(0);
            var dDone = workerStatus(d, 0).getFinishedTaskList(0);
            assertEquals(77, pDone.getBatchId());
            assertEquals(-1, dDone.getBatchId());
            assertEquals(300, dDone.getExecutionTimeMs());
            assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_PENDING, pDone.getPhase());
            assertTrue(pDone.getIsWaiting());
            assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_PENDING, dDone.getPhase());
            assertEquals(0, d.getOccupiedKvTokens());
        }
    }

    @Test void blockedAllocateIsReceivedWithoutExecutionSlotAndCancelSettlesOnce() throws Exception {
        var model = performanceModel(dir, "10", 1, 300);
        try (var c = MockEngineTestCluster.create(model, 63830, 1, 1)) {
            var p = c.prefill(0); var d = c.decode(0);
            var cache = (MockLruBlockCache) field(d, "cache");
            var held = cache.acquire(5836, List.of());
            assertNotNull(held);
            var ack = new CompletableFuture<EngineRpcService.EnqueueBatchResponsePB>();
            Thread.startVirtualThread(() -> {
                try { ack.complete(enqueue(p, batch(78, slot(0, inputWithDecode(2, 3072, d.getGrpcPort(), 2))))); }
                catch (Throwable error) { ack.completeExceptionally(error); }
            });
            try {
                await(() -> workerStatus(d, 0).getRunningTaskInfoCount() == 1);
                var task = workerStatus(d, 0).getRunningTaskInfo(0);
                assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED, task.getPhase());
                assertEquals(-1, task.getBatchId());
                assertTrue(task.getIsWaiting());
                assertFalse(ack.isDone());
                assertEquals(0, d.getActiveDecodeCount());
                assertEquals(0, d.whaleMetrics().get("rtp_llm_running_stream_size").intValue());
                assertEquals(0, d.whaleMetrics().get("rtp_llm_wait_stream_size").intValue());
                assertEquals(0, d.whaleMetrics().get("rtp_llm_loading_cache_stream_size").intValue());
                assertTrue(p.cancelRequest(2).found());
                assertEquals(8429, ack.get(3, TimeUnit.SECONDS).getErrors(0).getErrorInfo().getErrorCode());
                c.awaitAllInflightZero(2000);
                assertEquals(0, workerStatus(d, 0).getRunningTaskInfoCount());
                assertEquals(1, workerStatus(d, 0).getFinishedTaskListCount());
                assertEquals(8100, workerStatus(d, 0).getFinishedTaskList(0).getErrorInfo().getErrorCode());
            } finally { cache.release(held); }
        }
    }

    @Test void downstreamPriorityCancelFetchPreserves8429AndOneTerminal() throws Exception {
        var model = performanceModel(dir, "10", 1, 200, Map.of(), Map.of("tokens_per_step", 1));
        try (var c = MockEngineTestCluster.create(model, 63840, 1, 1)) {
            var p = c.prefill(0); var d = c.decode(0);
            p.setAutoFetch(false);
            assertEquals(1, enqueue(p, batch(79, slot(0, inputWithDecode(3, 1024, d.getGrpcPort(), 20)))).getSuccessesCount());
            var output = fetch(p, 3);
            await(() -> d.getActiveDecodeCount() == 1);
            assertTrue(p.cancelRequest(3).found());
            for (var task : workerStatus(p, 0).getRunningTaskInfoList())
                assertEquals(79, task.getBatchId(), "P overlay must retain P identity");
            var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> output.done.get(3, TimeUnit.SECONDS));
            assertEquals(8429, MockRpcErrors.downstream(failure.getCause()).getErrorCode());
            c.awaitAllInflightZero(2000);
            var terminals = workerStatus(p, 0).getFinishedTaskListList();
            assertEquals(1, terminals.stream().filter(t -> t.getErrorInfo().getErrorCode() == 8429).count());
            assertTrue(terminals.stream().allMatch(t -> t.getBatchId() == 79));
            assertEquals(0, workerStatus(p, 0).getRunningTaskInfoCount());
            assertEquals(0, p.getOccupiedKvTokens());
            assertEquals(0, d.getOccupiedKvTokens());
        }
    }

    @Test void insufficientGrowthFailsBeforeAnyDecodeForward() throws Exception {
        var model = performanceModel(dir, "10", 1, 300, Map.of(), Map.of("tokens_per_step", 1));
        model.nativeTokenCacheKeys = true;
        try (var c = MockEngineTestCluster.create(model, 63850, 1, 1)) {
            var p = c.prefill(0); var d = c.decode(0);
            p.setAutoFetch(false);
            assertEquals(1, enqueue(p, batch(80, slot(0, inputWithDecode(4, 1024, d.getGrpcPort(), 2)))).getSuccessesCount());
            c.awaitNoInflight(p, 2000);
            var cache = (MockLruBlockCache) field(d, "cache");
            var held = cache.acquire(cache.availableBlocks() - cache.reserveBlocks(), List.of());
            assertNotNull(held);
            assertTrue(cache.growTo(held, held.totalBlocks() + cache.availableBlocks()).success());
            var events = new CopyOnWriteArrayList<Map<String, Number>>();
            var reporter = d.getClass().getDeclaredField("eventMetricReporter");
            reporter.setAccessible(true);
            reporter.set(d, (java.util.function.Consumer<Map<String, Number>>) events::add);
            try {
                var output = fetch(p, 4);
                var failure = assertThrows(java.util.concurrent.ExecutionException.class,
                        () -> output.done.get(3, TimeUnit.SECONDS));
                assertEquals(602, MockRpcErrors.downstream(failure.getCause()).getErrorCode());
                c.awaitAllInflightZero(2000);
                assertEquals(0, d.getActiveDecodeCount());
                assertTrue(events.stream().noneMatch(event -> event.containsKey("mock_decode_step_forward_us")));
                assertEquals(602, workerStatus(d, 0).getFinishedTaskList(0).getErrorInfo().getErrorCode());
            } finally { cache.release(held); }
        }
    }

    private static Output fetch(JavaMockEngineCluster.FastRpcService p, long id) {
        var output = new Output();
        p.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder().setRequestId(id).build(), output);
        return output;
    }
    private static final class Output implements StreamObserver<EngineRpcService.GenerateOutputsPB> {
        final CompletableFuture<Void> done = new CompletableFuture<>();
        final List<EngineRpcService.GenerateOutputsPB> frames = new CopyOnWriteArrayList<>();
        public void onNext(EngineRpcService.GenerateOutputsPB output) { frames.add(output); }
        public void onError(Throwable error) { done.completeExceptionally(error); }
        public void onCompleted() { done.complete(null); }
    }
    private static Object field(Object object, String name) throws Exception {
        var field = object.getClass().getDeclaredField(name);
        field.setAccessible(true);
        return field.get(object);
    }
    private static void await(BooleanSupplier ready) throws Exception {
        long end = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (!ready.getAsBoolean() && System.nanoTime() < end) Thread.sleep(1);
        assertTrue(ready.getAsBoolean(), "transition timed out");
    }
}
