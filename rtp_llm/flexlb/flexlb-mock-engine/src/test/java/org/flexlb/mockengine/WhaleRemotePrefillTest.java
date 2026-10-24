package org.flexlb.mockengine;

import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Existing Prefill/Fetch flow with all Decode communication crossing a real gRPC socket. */
class WhaleRemotePrefillTest {

    private static final int BASE_PORT = Integer.parseInt(System.getenv().getOrDefault("FLEXLB_PORT_BASE", "62600"));

    @TempDir
    Path tempDir;

    private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(4);
    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services = new ConcurrentHashMap<>();
    private JavaMockEngineCluster.FastRpcService prefill;
    private JavaMockEngineCluster.FastRpcService decode;

    @AfterEach
    void tearDown() throws InterruptedException {
        var closes = new java.util.ArrayList<Runnable>();
        if (remoteServer != null) closes.add(() -> {
            remoteServer.shutdownNow();
            try { assertTrue(remoteServer.awaitTermination(3, TimeUnit.SECONDS)); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException(error); }
        });
        services.values().forEach(service -> closes.add(service::shutdown));
        closes.add(() -> {
            scheduler.shutdownNow();
            try { assertTrue(scheduler.awaitTermination(3, TimeUnit.SECONDS)); }
            catch (InterruptedException error) { Thread.currentThread().interrupt(); throw new IllegalStateException(error); }
        });
        ShutdownTasks.runAll(closes);
    }

    // ==================== multi-frame ttft < e2e ====================

    @Test
    @Timeout(30)
    void fetchResponseDeliversFirstTokenFrameThenTerminalFrame() throws Exception {
        startCluster("100", 50.0);
        long requestId = 1;

        EngineRpcService.EnqueueBatchResponsePB ack = enqueue(prefill,
                batch(7000, slot(0, inputWithDecode(requestId, 10, decode.getGrpcPort()))));
        assertEquals(1, ack.getSuccessesCount(), "request must be admitted");
        assertEquals(0, ack.getErrorsCount());

        CollectedStream stream = fetch(prefill, requestId, 10_000);

        assertNull(stream.error.get(), "stream must not error");
        assertEquals(8, stream.frames.size(),
                "fetch_response must stream every generated token, got " + stream.frames.size());
        assertFalseFrame(stream.frames.get(0), "frame 0 must be the prefill first-token frame");
        assertTrueFrame(stream.frames.get(7), "last frame must be terminal");
        assertFrontendTensor(stream.frames.get(0), 1, 9);
        for (var frame : stream.frames) assertFrontendTensor(frame, 1, 9);
        assertEquals(8, stream.frames.get(7).getFlattenOutput().getAuxInfo(0).getOutputLen());
        assertEquals(8, stream.frames.stream().mapToLong(frame ->
                frame.getFlattenOutput().getOutputIds().getShape(2)).sum(),
                "frontend concatenation must produce exactly max_new_tokens");
        assertTrue(stream.virtualThreads.stream().allMatch(Boolean::booleanValue),
                "Whale response waits must not occupy native threads");
        assertTtftStrictlyBeforeE2e(stream);
    }

    @Test @Timeout(10)
    void downstreamFailureCannotReleaseGpuWhilePrefillForwardStillRuns() throws Exception {
        startCluster("1000", 5.0);
        var field = prefill.getClass().getDeclaredField("performance"); field.setAccessible(true);
        var model = (MockPerformanceModel)field.get(prefill);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
        model.pdPolicy = new MockPdPolicy(0, 1000, 10000, 1, 5000, 10000, 128, 2000);
        var ack = enqueue(prefill, batch(7900, slot(0, inputWithDecode(7900, 10, decode.getGrpcPort()))));
        assertEquals(1, ack.getSuccessesCount(), ack.toString());
        assertTrue(prefill.getActiveKvTokens() > 0);
        remoteServer.shutdownNow();
        var stream = fetch(prefill, 7900, 2000);
        assertNotNull(stream.error.get());
        assertTrue(prefill.getActiveKvTokens() > 0, "transport failure must retain GPU ownership until the forward boundary");
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while ((prefill.getActiveKvTokens() > 0 || prefill.getInflightCount() > 0) && System.nanoTime() < until) Thread.sleep(1);
        assertEquals(0, prefill.getActiveKvTokens());
        assertEquals(0, prefill.getInflightCount());
    }

    private static void assertFrontendTensor(EngineRpcService.GenerateOutputsPB frame, int tokens, int token) {
        var output = frame.getFlattenOutput();
        assertTrue(output.hasOutputIds(), "normal frontend must receive output_ids");
        var tensor = output.getOutputIds();
        assertEquals(EngineRpcService.TensorPB.DataType.INT32, tensor.getDataType());
        assertEquals(List.of(1L, 1L, (long) tokens), tensor.getShapeList());
        assertEquals(tokens * Integer.BYTES, tensor.getInt32Data().size());
        var bytes = tensor.getInt32Data().asReadOnlyByteBuffer().order(java.nio.ByteOrder.LITTLE_ENDIAN);
        while (bytes.hasRemaining()) assertEquals(token, bytes.getInt());
        assertEquals(tokens, output.getAuxInfo(0).getStepOutputLen());
    }

    @Test
    @Timeout(30)
    void singleTokenRequestDoesNotRepeatPrefillTokenInDecodeTerminal() throws Exception {
        startCluster("10", 5.0);
        var input = inputWithDecode(77, 10, decode.getGrpcPort()).toBuilder();
        input.setGenerateConfig(input.getGenerateConfig().toBuilder().setMaxNewTokens(1));
        var ack = enqueue(prefill, batch(7077, slot(0, input.build())));
        assertEquals(1, ack.getSuccessesCount());
        var stream = fetch(prefill, 77, 10_000);
        assertNull(stream.error.get());
        assertEquals(1, stream.frames.size(), "frontend cannot decode an empty tensor frame");
        assertFrontendTensor(stream.frames.get(0), 1, 9);
        assertTrueFrame(stream.frames.get(0), "single token must finish in one nonempty frame");
        assertEquals(1, stream.frames.get(0).getFlattenOutput().getAuxInfo(0).getOutputLen());
    }

    @Test
    @Timeout(30)
    void generateStreamDeliversFirstTokenFrameThenTerminalFrame() throws Exception {
        startCluster("100", 50.0);
        long requestId = 2;

        CollectedStream stream = generate(prefill,
                inputWithDecode(requestId, 10, decode.getGrpcPort()), 10_000);

        assertNull(stream.error.get(), "stream must not error");
        assertEquals(8, stream.frames.size(),
                "generate_stream must stream every generated token, got " + stream.frames.size());
        assertFalseFrame(stream.frames.get(0), "frame 0 must be the prefill first-token frame");
        assertTrueFrame(stream.frames.get(7), "last frame must be terminal");
        assertTrue(stream.virtualThreads.stream().allMatch(Boolean::booleanValue),
                "Whale response waits must not occupy native threads");
        assertTtftStrictlyBeforeE2e(stream);
    }

    // ==================== timeout semantics ====================

    @Test
    @Timeout(30)
    void decodeProgressKeepsFetchAlivePastIdleTimeout() throws Exception {
        // Total decode time exceeds the idle timeout, but progress frames
        // keep the stream alive through its terminal output.
        startCluster("100", 50.0);
        prefill.setResponsePollTimeoutMs(150);
        decode.setResponsePollTimeoutMs(150);
        long requestId = 3;

        EngineRpcService.EnqueueBatchResponsePB ack = enqueue(prefill,
                batch(7001, slot(0, inputWithDecode(requestId, 10, decode.getGrpcPort()))));
        assertEquals(1, ack.getSuccessesCount());

        CollectedStream stream = fetch(prefill, requestId, 10_000);

        assertNull(stream.error.get(), "progress must keep Fetch alive");
        assertEquals(8, stream.frames.size());
        assertTrueFrame(stream.frames.get(7), "must finish despite total time exceeding idle timeout");
    }

    @Test
    @Timeout(30)
    void unknownFetchReturnsNotFoundWithoutCreatingContext() throws Exception {
        startCluster("100", 50.0);
        prefill.setResponsePollTimeoutMs(150);
        decode.setResponsePollTimeoutMs(150);

        // Real Fetch takes an existing deferred context once; an unknown ID
        // fails immediately rather than allocating an orphan response queue.
        CollectedStream stream = fetch(prefill, 4, 10_000);

        assertEquals(io.grpc.Status.Code.NOT_FOUND,
                io.grpc.Status.fromThrowable(stream.error.get()).getCode());
        assertEquals(0, stream.frames.size(), "no frame may be delivered for an unknown request");
    }

    // ==================== assertions ====================

    private static void assertFalseFrame(EngineRpcService.GenerateOutputsPB frame, String message) {
        assertEquals(1, frame.getFlattenOutput().getFinishedCount(), message);
        assertEquals(false, frame.getFlattenOutput().getFinished(0), message);
    }

    private static void assertTrueFrame(EngineRpcService.GenerateOutputsPB frame, String message) {
        assertEquals(1, frame.getFlattenOutput().getFinishedCount(), message);
        assertEquals(true, frame.getFlattenOutput().getFinished(0), message);
    }

    private static void assertTtftStrictlyBeforeE2e(CollectedStream stream) {
        assertNotNull(stream.frameNanos.get(0), "first frame timestamp");
        assertNotNull(stream.frameNanos.get(1), "terminal frame timestamp");
        long ttftNanos = stream.frameNanos.get(0);
        long e2eNanos = stream.frameNanos.get(1);
        // Guard against scheduling noise: decode is 400 ms, so the gap must be
        // at least 100 ms — far above any timer resolution ambiguity.
        long gapMs = TimeUnit.NANOSECONDS.toMillis(e2eNanos - ttftNanos);
        assertTrue(e2eNanos > ttftNanos,
                "terminal frame must be strictly AFTER first-token frame (ttft < e2e), gap=" + gapMs + "ms");
        assertTrue(gapMs >= 100,
                "frame gap must reflect real decode execution (~400ms), got " + gapMs + "ms");
    }

    // ==================== cluster / RPC helpers ====================

    private io.grpc.Server remoteServer;

    private void startCluster(String prefillFormulaMs, double decodeStepMs) throws IOException {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        new com.fasterxml.jackson.databind.ObjectMapper().writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "jitter_pct", 0.0,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0,
                        "tokens_per_step", 1.0,
                        "step_ms_by_batch", List.of(List.of(1, decodeStepMs)))));
        MockMasterConfig.writeWithPrefillExpression(master, prefillFormulaMs);
        MockPerformanceModel model = MockModelFixture.load(performance.toString(), master.toString());

        prefill = new JavaMockEngineCluster.FastRpcService(
                "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                BASE_PORT, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats());
        services.put(BASE_PORT, prefill);
        decode = new JavaMockEngineCluster.FastRpcService(
                "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                BASE_PORT + 1, services, scheduler, model, 100,
                new JavaMockEngineCluster.ClusterStats());
        services.put(BASE_PORT + 1, decode);
        prefill.setWhaleRemote(true);
        decode.setWhaleRemote(true);
        remoteServer = io.grpc.ServerBuilder.forPort(BASE_PORT + 1).addService(decode).build().start();
    }

    private static EngineRpcService.GenerateInputPB inputWithDecode(
            long requestId, int inputTokens, int decodePort) {
        EngineRpcService.GenerateInputPB.Builder input = EngineRpcService.GenerateInputPB.newBuilder()
                .setRequestId(requestId)
                .setGenerateConfig(EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(8)
                        .addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                                .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                .setRoleStr("DECODE")
                                .setIp("127.0.0.1").setGrpcPort(decodePort)
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

    private static EngineRpcService.EnqueueBatchResponsePB enqueue(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.EnqueueBatchRequestPB request) {
        AtomicReference<EngineRpcService.EnqueueBatchResponsePB> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        service.enqueueBatch(request, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.EnqueueBatchResponsePB value) {
                response.set(value);
            }

            @Override
            public void onError(Throwable t) {
                error.set(t);
                latch.countDown();
            }

            @Override
            public void onCompleted() {
                latch.countDown();
            }
        });
        try {
            assertTrue(latch.await(10, TimeUnit.SECONDS), "enqueueBatch must ack");
            assertNull(error.get(), "enqueueBatch must not error");
            assertNotNull(response.get(), "enqueueBatch response");
            return response.get();
        } catch (InterruptedException e) {
            throw new AssertionError("interrupted waiting for enqueueBatch ack", e);
        }
    }

    private record CollectedStream(List<EngineRpcService.GenerateOutputsPB> frames,
                                   List<Long> frameNanos,
                                   List<Boolean> virtualThreads,
                                   AtomicReference<Throwable> error) {
    }

    private static CollectedStream collect(Consumer<StreamObserver<EngineRpcService.GenerateOutputsPB>> invoke,
                                           long awaitMs) throws InterruptedException {
        List<EngineRpcService.GenerateOutputsPB> frames = new CopyOnWriteArrayList<>();
        List<Long> frameNanos = new CopyOnWriteArrayList<>();
        List<Boolean> virtualThreads = new CopyOnWriteArrayList<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch terminal = new CountDownLatch(1);
        invoke.accept(new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
                frames.add(value);
                frameNanos.add(System.nanoTime());
                virtualThreads.add(Thread.currentThread().isVirtual());
            }

            @Override
            public void onError(Throwable t) {
                error.set(t);
                terminal.countDown();
            }

            @Override
            public void onCompleted() {
                terminal.countDown();
            }
        });
        assertTrue(terminal.await(awaitMs, TimeUnit.MILLISECONDS),
                "stream must terminate (completed or error) within " + awaitMs + "ms");
        return new CollectedStream(new ArrayList<>(frames), new ArrayList<>(frameNanos),
                new ArrayList<>(virtualThreads), error);
    }

    private static CollectedStream fetch(JavaMockEngineCluster.FastRpcService service,
                                         long requestId, long awaitMs) throws InterruptedException {
        return collect(observer -> service.fetchResponse(
                EngineRpcService.FetchRequestPB.newBuilder().setRequestId(requestId).build(),
                observer), awaitMs);
    }

    private static CollectedStream generate(JavaMockEngineCluster.FastRpcService service,
                                            EngineRpcService.GenerateInputPB input,
                                            long awaitMs) throws InterruptedException {
        return collect(observer -> service.generateStreamCall(input, observer), awaitMs);
    }
    @Test @Timeout(15)
    void priorityCancelInterruptsRemoteAllocateUnderKvPressure() throws Exception {
        startCluster("100", 5);
        var field = prefill.getClass().getDeclaredField("performance"); field.setAccessible(true);
        var model = (MockPerformanceModel)field.get(prefill);
        model.setPrefillBatchPolicy(new MockPrefillBatchPolicy(4, 10000, 10000, 1, false, 0, 0, false));
        model.pdPolicy = new MockPdPolicy(0, 5000, 10000, 1, 5000, 10000, 128, 2000);
        field = decode.getClass().getDeclaredField("cache"); field.setAccessible(true);
        var pool = (MockLruBlockCache)field.get(decode);
        var held = pool.acquire(pool.totalBlocks() - pool.reserveBlocks(), List.of());
        org.junit.jupiter.api.Assertions.assertNotNull(held);
        try {
            var response = java.util.concurrent.CompletableFuture.supplyAsync(() -> enqueue(prefill,
                    batch(7990, slot(0, inputWithDecode(7990, 10, decode.getGrpcPort())))));
            field = prefill.getClass().getDeclaredField("remotePrefillOwners"); field.setAccessible(true);
            var owners = (Map<?, ?>)field.get(prefill);
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while (!owners.containsKey(7990L) && System.nanoTime() < deadline) Thread.sleep(1);
            assertTrue(owners.containsKey(7990L));
            assertTrue(prefill.cancelRequest(7990).found());
            var ack = response.get(1, TimeUnit.SECONDS);
            assertEquals(0, ack.getSuccessesCount()); assertEquals(1, ack.getErrorsCount());
            assertEquals(8429, ack.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, decode.getAcceptedCount()); assertEquals(0, prefill.getOccupiedKvTokens());
        } finally { pool.release(held); }
    }

}
