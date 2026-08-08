package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Long-poll behaviors of {@code getWorkerStatus} (scheduler upgrade A):
 * park while no new completion, wake on the next completion event, timeout
 * fallback, immediate answer on stop/shutdown, and full degradation when
 * {@code wait_timeout_ms} is 0/absent. The gRPC server runs on a
 * directExecutor, so all of these must complete without blocking the caller
 * thread — verified here by invoking the handler inline and observing the
 * response asynchronously.
 */
class MockEngineStatusLongPollTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);

    @AfterEach
    void tearDown() throws InterruptedException {
        scheduler.shutdownNow();
        scheduler.awaitTermination(2, TimeUnit.SECONDS);
    }

    @Test
    void longPollParksWhileNoNewCompletionExists() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("60", 1.0));

        CompletableFuture<EngineRpcService.WorkerStatusPB> poll = longPoll(service, 0, 5_000);

        // The handler returned immediately (non-blocking) but the response must
        // stay parked while no completion newer than version 0 exists.
        Thread.sleep(150);
        assertFalse(poll.isDone(), "long-poll must stay parked without a new completion");

        // Release the waiter so the test does not leak a parked observer.
        service.drainAndShutdown();
        poll.get(2, TimeUnit.SECONDS);
    }

    @Test
    void longPollWakesOnNextCompletionEvent() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("60", 1.0));

        CompletableFuture<EngineRpcService.WorkerStatusPB> poll = longPoll(service, 0, 5_000);
        long enqueuedAt = System.nanoTime();
        enqueue(service, batch(1, slot(0, input(1, 100))));

        EngineRpcService.WorkerStatusPB status = poll.get(2, TimeUnit.SECONDS);
        long waitedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - enqueuedAt);

        assertEquals(1, status.getFinishedTaskListCount(),
                "woken poll must carry the completion that triggered it");
        assertTrue(status.getLatestFinishedVersion() > 0);
        assertTrue(waitedMs < 2_000,
                "poll must return on the completion event, not the 5s timeout, waited " + waitedMs + "ms");
    }

    @Test
    void longPollTimesOutWithoutCompletion() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("60", 1.0));

        long start = System.nanoTime();
        CompletableFuture<EngineRpcService.WorkerStatusPB> poll = longPoll(service, 0, 150);

        EngineRpcService.WorkerStatusPB status = poll.get(2, TimeUnit.SECONDS);
        long waitedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);

        assertEquals(0, status.getFinishedTaskListCount());
        assertEquals(0, status.getLatestFinishedVersion());
        assertTrue(status.getAlive());
        assertTrue(waitedMs >= 100, "timeout response fired too early: " + waitedMs + "ms");
    }

    @Test
    void longPollReturnsImmediatelyOnShutdown() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("60", 1.0));

        CompletableFuture<EngineRpcService.WorkerStatusPB> poll = longPoll(service, 0, 10_000);
        Thread.sleep(50);
        assertFalse(poll.isDone());

        service.drainAndShutdown();

        EngineRpcService.WorkerStatusPB status = poll.get(1, TimeUnit.SECONDS);
        assertFalse(status.getAlive(), "shutdown flush must report alive=false");
    }

    @Test
    void longPollReturnsImmediatelyOnStopEngine() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("60", 1.0));

        CompletableFuture<EngineRpcService.WorkerStatusPB> poll = longPoll(service, 0, 10_000);
        Thread.sleep(50);
        assertFalse(poll.isDone());

        service.setStopped(true);

        EngineRpcService.WorkerStatusPB status = poll.get(1, TimeUnit.SECONDS);
        assertFalse(status.getAlive(), "stop_engine flush must report alive=false");

        // A stopped engine must answer subsequent long-polls immediately.
        EngineRpcService.WorkerStatusPB direct = longPoll(service, 0, 10_000)
                .get(1, TimeUnit.SECONDS);
        assertFalse(direct.getAlive());
    }

    @Test
    void zeroWaitTimeoutKeepsImmediateResponse() throws Exception {
        JavaMockEngineCluster.FastRpcService service = service(model("60", 1.0));

        // wait_timeout_ms absent (0) with no completions: original behavior,
        // the response is delivered synchronously within the handler call.
        CompletableFuture<EngineRpcService.WorkerStatusPB> poll = longPoll(service, 0, 0);
        assertTrue(poll.isDone(), "wait_timeout_ms=0 must keep the immediate response path");
        assertEquals(0, poll.get().getFinishedTaskListCount());
        assertTrue(poll.get().getAlive());
    }

    private static CompletableFuture<EngineRpcService.WorkerStatusPB> longPoll(
            JavaMockEngineCluster.FastRpcService service, long sinceVersion, long waitTimeoutMs) {
        CompletableFuture<EngineRpcService.WorkerStatusPB> future = new CompletableFuture<>();
        service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(sinceVersion)
                        .setWaitTimeoutMs(waitTimeoutMs)
                        .build(),
                new StreamObserver<>() {
                    @Override
                    public void onNext(EngineRpcService.WorkerStatusPB value) {
                        future.complete(value);
                    }

                    @Override
                    public void onError(Throwable throwable) {
                        future.completeExceptionally(throwable);
                    }

                    @Override
                    public void onCompleted() {
                    }
                });
        return future;
    }

    private JavaMockEngineCluster.FastRpcService service(MockPerformanceModel model) {
        Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
        JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                "prefill",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                61_100,
                services,
                scheduler,
                model,
                100,
                new JavaMockEngineCluster.ClusterStats());
        services.put(61_100, service);
        return service;
    }

    private MockPerformanceModel model(String formula, double sleepScale) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", sleepScale,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0, "step_ms_by_batch", List.of(List.of(1, 1.0)))));
        MAPPER.writeValue(master.toFile(), Map.of(
                "zone_process_setting", Map.of(
                        "process_info", Map.of(
                                "envs", List.of(List.of("PREFILL_TIME_FORMULA", formula))))));
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

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

    private static void enqueue(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.EnqueueBatchRequestPB request) {
        CompletableFuture<EngineRpcService.EnqueueBatchResponsePB> ack = new CompletableFuture<>();
        service.enqueueBatch(request, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.EnqueueBatchResponsePB value) {
                ack.complete(value);
            }

            @Override
            public void onError(Throwable throwable) {
                ack.completeExceptionally(throwable);
            }

            @Override
            public void onCompleted() {
            }
        });
        assertTrue(ack.isDone(), "enqueue ack is synchronous");
    }
}
