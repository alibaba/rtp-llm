package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

/** Lifecycle fixture for the standard in-process prefill/decode test cluster. */
final class MockEngineTestCluster implements AutoCloseable {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private final ScheduledExecutorService scheduler;
    private final Map<Integer, JavaMockEngineCluster.FastRpcService> services =
            new ConcurrentHashMap<>();
    private final List<JavaMockEngineCluster.FastRpcService> prefills = new ArrayList<>();
    private final List<JavaMockEngineCluster.FastRpcService> decodes = new ArrayList<>();
    private MockControlServer controlServer;

    private MockEngineTestCluster(
            MockPerformanceModel model,
            int basePort,
            int prefillCount,
            int decodeCount,
            int schedulerThreads,
            boolean startControlServer) throws IOException {
        scheduler = Executors.newScheduledThreadPool(schedulerThreads, runnable -> {
            Thread thread = new Thread(runnable, "mock-engine-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        addEngines(model, basePort, prefillCount, decodeCount);
        if (startControlServer) {
            controlServer = new MockControlServer(
                    services, new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
            controlServer.start();
        }
    }

    static MockEngineTestCluster start(
            MockPerformanceModel model, int basePort, int prefillCount, int decodeCount)
            throws IOException {
        return start(model, basePort, prefillCount, decodeCount, 8);
    }

    static MockEngineTestCluster start(
            MockPerformanceModel model,
            int basePort,
            int prefillCount,
            int decodeCount,
            int schedulerThreads) throws IOException {
        return new MockEngineTestCluster(
                model, basePort, prefillCount, decodeCount, schedulerThreads, true);
    }

    static MockEngineTestCluster create(
            MockPerformanceModel model, int basePort, int prefillCount, int decodeCount)
            throws IOException {
        return create(model, basePort, prefillCount, decodeCount, 8);
    }

    static MockEngineTestCluster create(
            MockPerformanceModel model,
            int basePort,
            int prefillCount,
            int decodeCount,
            int schedulerThreads) throws IOException {
        return new MockEngineTestCluster(
                model, basePort, prefillCount, decodeCount, schedulerThreads, false);
    }

    private void addEngines(
            MockPerformanceModel model, int basePort, int prefillCount, int decodeCount) {
        for (int i = 0; i < prefillCount; i++) {
            prefills.add(addEngine(
                    model,
                    basePort + i,
                    "prefill",
                    EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL));
        }
        for (int i = 0; i < decodeCount; i++) {
            decodes.add(addEngine(
                    model,
                    basePort + prefillCount + i,
                    "decode",
                    EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE));
        }
    }

    private JavaMockEngineCluster.FastRpcService addEngine(
            MockPerformanceModel model,
            int port,
            String role,
            EngineRpcService.RoleTypePB roleType) {
        JavaMockEngineCluster.FastRpcService service =
                new JavaMockEngineCluster.FastRpcService(
                        role,
                        roleType,
                        port,
                        services,
                        scheduler,
                        model,
                        100,
                        new JavaMockEngineCluster.ClusterStats());
        services.put(port, service);
        return service;
    }

    Map<Integer, JavaMockEngineCluster.FastRpcService> services() {
        return services;
    }

    List<JavaMockEngineCluster.FastRpcService> prefills() {
        return prefills;
    }

    List<JavaMockEngineCluster.FastRpcService> decodes() {
        return decodes;
    }

    JavaMockEngineCluster.FastRpcService prefill(int index) {
        return prefills.get(index);
    }

    JavaMockEngineCluster.FastRpcService decode(int index) {
        return decodes.get(index);
    }

    int controlPort() {
        if (controlServer == null) {
            throw new IllegalStateException("control server was not requested");
        }
        return controlServer.getPort();
    }

    JsonNode snapshot() throws IOException, InterruptedException {
        return MAPPER.readTree(MockEngineTestSupport.httpGet(controlPort(), "/snapshot"))
                .path("engines");
    }

    long totalCompleted() {
        return services.values().stream()
                .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                .sum();
    }

    void enqueueBatch(
            JavaMockEngineCluster.FastRpcService prefill,
            long batchId,
            int startRequestId,
            int count,
            List<JavaMockEngineCluster.FastRpcService> decodeEngines) {
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[count];
        for (int i = 0; i < count; i++) {
            int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
            inputs[i] = MockEngineTestSupport.inputWithDecode(startRequestId + i, 10, decodePort);
        }
        MockEngineTestSupport.enqueue(
                prefill,
                MockEngineTestSupport.batch(batchId, MockEngineTestSupport.slot(0, inputs)));
    }

    void awaitCompleted(int expected, long timeoutMs) throws InterruptedException {
        await(() -> totalCompleted() >= expected, timeoutMs,
                () -> "expected " + expected + " completions, got " + totalCompleted());
    }

    void awaitInflight(
            JavaMockEngineCluster.FastRpcService service, int minimum, long timeoutMs)
            throws InterruptedException {
        await(() -> service.getInflightCount() >= minimum, timeoutMs,
                () -> "inflight never reached " + minimum + " on port "
                        + service.getGrpcPort() + ", got " + service.getInflightCount());
    }

    void awaitNoInflight(JavaMockEngineCluster.FastRpcService service, long timeoutMs)
            throws InterruptedException {
        await(() -> service.getInflightCount() == 0, timeoutMs,
                () -> "inflight never reached zero on port " + service.getGrpcPort());
    }

    void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        await(() -> services.values().stream()
                        .allMatch(service -> service.getInflightCount() == 0),
                timeoutMs, this::inflightSummary);
    }

    void assertAllInflightZero() {
        if (services.values().stream().anyMatch(service -> service.getInflightCount() != 0)) {
            throw new AssertionError(inflightSummary());
        }
    }

    private static void await(
            BooleanSupplier condition,
            long timeoutMs,
            java.util.function.Supplier<String> timeoutMessage) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(5);
        }
        throw new AssertionError(timeoutMessage.get());
    }

    private String inflightSummary() {
        StringBuilder result = new StringBuilder("inflight not zero:");
        services.values().forEach(service -> result.append(" port=")
                .append(service.getGrpcPort())
                .append(" inflight=")
                .append(service.getInflightCount())
                .append(" running=")
                .append(service.getRunningCount()));
        return result.toString();
    }

    @Override
    public void close() {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        services.values().forEach(JavaMockEngineCluster.FastRpcService::shutdown);
        scheduler.shutdownNow();
        try {
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
