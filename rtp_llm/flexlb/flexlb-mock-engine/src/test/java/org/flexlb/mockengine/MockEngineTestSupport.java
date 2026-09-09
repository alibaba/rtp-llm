package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/** Shared protocol primitives for in-process mock-engine tests. */
final class MockEngineTestSupport {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();

    private MockEngineTestSupport() {
    }

    static EngineRpcService.GenerateInputPB input(long requestId, int inputTokens) {
        return input(requestId, inputTokens, 1, null);
    }

    static EngineRpcService.GenerateInputPB inputWithDecode(
            long requestId, int inputTokens, int decodePort) {
        return input(requestId, inputTokens, 1, decodePort);
    }

    static EngineRpcService.GenerateInputPB inputWithDecode(
            long requestId, int inputTokens, int decodePort, int outputTokens) {
        return input(requestId, inputTokens, outputTokens, decodePort);
    }

    /**
     * Input carrying hash-channel block keys: the keys travel in the
     * unique_key JSON ("block_cache_keys") exactly like the load client's
     * traffic — shape() smuggles them out of the metadata channel the master
     * never inspects (the data-link analysis' second channel).
     */
    static EngineRpcService.GenerateInputPB inputWithBlockKeys(
            long requestId, int inputTokens, List<Long> blockKeys) {
        String uniqueKey;
        try {
            uniqueKey = MAPPER.writeValueAsString(Map.of(
                    "input_len", inputTokens,
                    "block_cache_keys", blockKeys));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        EngineRpcService.GenerateConfigPB.Builder config =
                EngineRpcService.GenerateConfigPB.newBuilder()
                        .setMaxNewTokens(1)
                        .setUniqueKey(uniqueKey);
        EngineRpcService.GenerateInputPB.Builder input =
                EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(requestId)
                        .setGenerateConfig(config.build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    private static EngineRpcService.GenerateInputPB input(
            long requestId, int inputTokens, int outputTokens, Integer decodePort) {
        EngineRpcService.GenerateConfigPB.Builder config =
                EngineRpcService.GenerateConfigPB.newBuilder().setMaxNewTokens(outputTokens);
        if (decodePort != null) {
            config.addRoleAddrs(EngineRpcService.RoleAddrPB.newBuilder()
                    .setRole(EngineRpcService.RoleAddrPB.RoleType.DECODE)
                    .setRoleStr("DECODE")
                    .setGrpcPort(decodePort)
                    .build());
        }
        EngineRpcService.GenerateInputPB.Builder input =
                EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(requestId)
                        .setGenerateConfig(config.build());
        for (int token = 0; token < inputTokens; token++) {
            input.addTokenIds(token);
        }
        return input.build();
    }

    static EngineRpcService.EnqueueBatchDpSlotPB slot(
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

    static EngineRpcService.EnqueueBatchRequestPB batch(
            long batchId, EngineRpcService.EnqueueBatchDpSlotPB... slots) {
        return EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                .setBatchId(batchId)
                .addAllDpSlots(List.of(slots))
                .build();
    }

    static EngineRpcService.EnqueueBatchResponsePB enqueue(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.EnqueueBatchRequestPB request) {
        return unary(observer -> service.enqueueBatch(request, observer));
    }

    static EngineRpcService.WorkerStatusPB workerStatus(
            JavaMockEngineCluster.FastRpcService service, long sinceVersion) {
        return unary(observer -> service.getWorkerStatus(
                EngineRpcService.StatusVersionPB.newBuilder()
                        .setLatestFinishedVersion(sinceVersion)
                        .build(),
                observer));
    }

    static MockPerformanceModel performanceModel(Path tempDir, String prefillFormula)
            throws IOException {
        return performanceModel(tempDir, prefillFormula, 1.0, 1.0, Map.of(), Map.of());
    }

    static MockPerformanceModel performanceModel(
            Path tempDir, String prefillFormula, double sleepScale) throws IOException {
        return performanceModel(
                tempDir, prefillFormula, sleepScale, 1.0, Map.of(), Map.of());
    }

    static MockPerformanceModel performanceModel(
            Path tempDir, String prefillFormula, double sleepScale, double decodeStepMs)
            throws IOException {
        return performanceModel(
                tempDir, prefillFormula, sleepScale, decodeStepMs, Map.of(), Map.of());
    }

    static MockPerformanceModel performanceModel(
            Path tempDir,
            String prefillFormula,
            double sleepScale,
            double decodeStepMs,
            Map<String, ?> prefillOverrides,
            Map<String, ?> decodeOverrides) throws IOException {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        Map<String, Object> prefill = new LinkedHashMap<>();
        prefill.put("scale", 1.0);
        prefill.putAll(prefillOverrides);
        Map<String, Object> decode = new LinkedHashMap<>();
        decode.put("scale", 1.0);
        decode.put("step_ms_by_batch", List.of(List.of(1, decodeStepMs)));
        decode.putAll(decodeOverrides);
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", sleepScale,
                "jitter_pct", 0.0,
                "prefill", prefill,
                "decode", decode));
        MockMasterConfig.writeWithPrefillExpression(master, prefillFormula);
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    static MockPerformanceModel decodeModel(
            Path tempDir, double decodeStepMs, Integer maxPendingRequests)
            throws IOException {
        return decodeModel(tempDir, decodeStepMs, maxPendingRequests, false);
    }

    static MockPerformanceModel decodeModel(
            Path tempDir,
            double decodeStepMs,
            Integer maxPendingRequests,
            boolean reportQueuedAsKvAllocated) throws IOException {
        Map<String, Object> decodeConfig = new LinkedHashMap<>();
        if (maxPendingRequests != null) {
            decodeConfig.put("max_pending_requests", maxPendingRequests);
        }
        if (reportQueuedAsKvAllocated) {
            decodeConfig.put("report_queued_as_kv_allocated", true);
        }
        return performanceModel(
                tempDir, "10", 0.1, decodeStepMs, Map.of(), decodeConfig);
    }

    static String httpGet(int port, String path) throws IOException, InterruptedException {
        HttpResponse<String> response = HTTP_CLIENT.send(
                HttpRequest.newBuilder()
                        .uri(URI.create("http://127.0.0.1:" + port + path))
                        .GET()
                        .build(),
                HttpResponse.BodyHandlers.ofString());
        requireHttpOk(response, "GET " + path);
        return response.body();
    }

    static String httpPost(int port, String path, String body)
            throws IOException, InterruptedException {
        HttpResponse<String> response = httpPostResponse(port, path, body);
        requireHttpOk(response, "POST " + path);
        return response.body();
    }

    static HttpResponse<String> httpPostResponse(int port, String path, String body)
            throws IOException, InterruptedException {
        return HTTP_CLIENT.send(
                HttpRequest.newBuilder()
                        .uri(URI.create("http://127.0.0.1:" + port + path))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private static void requireHttpOk(HttpResponse<?> response, String operation) {
        if (response.statusCode() != 200) {
            throw new AssertionError(operation + " failed: HTTP " + response.statusCode());
        }
    }

    static MockPerformanceModel.RequestShape requestShape(
            MockPerformanceModel model, long requestId, int inputTokens) {
        return model.shape(input(requestId, inputTokens), new MockLruBlockCache(100));
    }

    static JavaMockEngineCluster.FastRpcService decodeService(
            MockPerformanceModel model,
            int port,
            Map<Integer, JavaMockEngineCluster.FastRpcService> services,
            ScheduledExecutorService scheduler,
            int maxConcurrency) {
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
                        100,
                        new JavaMockEngineCluster.ClusterStats(),
                        10_000_000L,
                        maxConcurrency);
        services.put(port, service);
        return service;
    }

    static void awaitDecodeQuiescence(
            JavaMockEngineCluster.FastRpcService service, long timeoutMs)
            throws ReflectiveOperationException, InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (service.getInflightCount() == 0
                    && service.getRunningCount() == 0
                    && activeDecodeRequests(service) == 0) {
                return;
            }
            Thread.sleep(10);
        }
        throw new AssertionError(
                "engine did not quiesce: inflight=" + service.getInflightCount()
                        + " running=" + service.getRunningCount()
                        + " activeDecode=" + activeDecodeRequests(service)
                        + " kv=" + service.getActiveKvTokens());
    }

    static AtomicInteger activeDecodeRequestsRef(
            JavaMockEngineCluster.FastRpcService service) throws ReflectiveOperationException {
        Field field = JavaMockEngineCluster.FastRpcService.class
                .getDeclaredField("activeDecodeRequests");
        field.setAccessible(true);
        return (AtomicInteger) field.get(service);
    }

    static int activeDecodeRequests(JavaMockEngineCluster.FastRpcService service)
            throws ReflectiveOperationException {
        return activeDecodeRequestsRef(service).get();
    }

    static int decodePendingQueueSize(JavaMockEngineCluster.FastRpcService service)
            throws ReflectiveOperationException {
        Field field = JavaMockEngineCluster.FastRpcService.class
                .getDeclaredField("decodePendingQueue");
        field.setAccessible(true);
        return ((ArrayDeque<?>) field.get(service)).size();
    }

    static boolean scheduleDecodeCompletion(
            JavaMockEngineCluster.FastRpcService service,
            MockPerformanceModel.RequestShape shape,
            long batchId,
            LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB> responseQueue)
            throws ReflectiveOperationException {
        Method method = JavaMockEngineCluster.FastRpcService.class.getDeclaredMethod(
                "scheduleDecodeCompletion",
                MockPerformanceModel.RequestShape.class,
                long.class,
                LinkedBlockingQueue.class);
        method.setAccessible(true);
        return (Boolean) method.invoke(service, shape, batchId, responseQueue);
    }

    static <T> T unary(Consumer<StreamObserver<T>> invocation) {
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
                throw new AssertionError("unary response timeout");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError("interrupted waiting for unary response", e);
        }
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        if (response.get() == null) {
            throw new AssertionError("unary response missing");
        }
        return response.get();
    }
}
