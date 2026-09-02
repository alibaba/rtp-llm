package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;
import java.util.function.Predicate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Dynamic engine scale-out/in via the {@code /add_engine} + {@code /remove_engine}
 * control endpoints backed by {@link DynamicEngineManager}, with the discovery
 * file ({@link DiscoveryFileStore}) kept in sync for a file-discovery master.
 *
 * <p>Covers the four required behaviours:
 * <ol>
 *   <li>add → new gRPC port reachable, engine visible in /snapshot, entry
 *       present in the discovery file (HTTP port = grpc − 1);</li>
 *   <li>remove → port refuses connections, discovery entry gone, services map
 *       free of residue, and the response reports the engine's in-flight
 *       counters at removal time;</li>
 *   <li>concurrent add×N + remove×M crossfire → the discovery file always
 *       parses completely and its entry set equals the services map;</li>
 *   <li>a dynamically added engine serves the full request pipeline
 *       (enqueue → decode scheduling → generated output).</li>
 * </ol>
 */
class DynamicEngineScaleTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final AtomicInteger PORT_ALLOCATOR = new AtomicInteger(63200);

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private EventLoopGroup bossGroup;
    private EventLoopGroup workerGroup;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private Map<Integer, Server> serversByPort;
    private DynamicEngineManager engineManager;
    private DiscoveryFileStore discoveryFileStore;
    private Path discoveryFile;

    @AfterEach
    void tearDown() throws InterruptedException {
        if (controlServer != null) {
            controlServer.stop();
            controlServer = null;
        }
        if (serversByPort != null) {
            for (Server server : serversByPort.values()) {
                server.shutdownNow();
            }
            serversByPort = null;
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
            services = null;
        }
        if (bossGroup != null) {
            bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
            bossGroup = null;
        }
        if (workerGroup != null) {
            workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
            workerGroup = null;
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
            scheduler = null;
        }
        engineManager = null;
        discoveryFileStore = null;
    }

    // ════════════════════════════════════════════════════════════════
    //  Tests
    // ════════════════════════════════════════════════════════════════

    @Test
    void addEngineExposesNewGrpcPortSnapshotAndDiscoveryEntry() throws Exception {
        int basePort = startCluster(model("10", 1.0), 1, 1);

        JsonNode added = postOk("/add_engine", "{\"role\":\"decode\"}");
        assertEquals("ok", added.path("status").asText());
        assertEquals("added", added.path("action").asText());
        // Auto-allocated port = current max + 1.
        assertEquals(basePort + 2, added.path("port").asInt());
        assertEquals(basePort + 1, added.path("http_port").asInt());
        String engineName = added.path("engine").asText();

        // New port is gRPC-reachable and reports the added role.
        EngineRpcService.WorkerStatusPB status = grpcWorkerStatus(basePort + 2);
        // Local intake: WorkerStatusPB.getRole() carries the master-side
        // "RoleType.<role>" convention; assert the typed role field instead.
        assertEquals(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE, status.getRoleType());
        assertTrue(status.getAlive());

        // /snapshot contains the new engine.
        boolean snapshotHasNewEngine = false;
        for (JsonNode engine : getJson("/snapshot").path("engines")) {
            if (engineName.equals(engine.path("name").asText())
                    && ("127.0.0.1:" + (basePort + 2)).equals(engine.path("grpc_addr").asText())) {
                snapshotHasNewEngine = true;
            }
        }
        assertTrue(snapshotHasNewEngine, "new engine " + engineName + " missing from /snapshot");

        // Discovery file contains the new entry (HTTP port) and both decode hosts.
        List<String> decodeHosts = hostList(readDiscoveryFile(), "mock.decode.hosts.address");
        assertEquals(List.of("127.0.0.1:" + basePort, "127.0.0.1:" + (basePort + 1)), decodeHosts);
        assertEquals(1, hostList(readDiscoveryFile(), "mock.prefill.hosts.address").size());
    }

    @Test
    void removeEngineClosesPortAndStripsDiscoveryAndServicesEntries() throws Exception {
        int basePort = startCluster(model("10", 1.0), 1, 1);
        int victimPort = basePort + 1; // decode-0

        JsonNode removed = postOk("/remove_engine", "{\"port\":" + victimPort + "}");
        assertEquals("ok", removed.path("status").asText());
        assertEquals("removed", removed.path("action").asText());
        assertEquals(victimPort, removed.path("port").asInt());
        assertTrue(removed.has("running_at_removal"), "response must report running_at_removal");
        assertTrue(removed.has("waiting_at_removal"), "response must report waiting_at_removal");

        // Port no longer accepts connections (server listener closed).
        awaitPortRefused(victimPort);

        // No residue in the services map or the server map.
        assertFalse(services.containsKey(victimPort), "services map still holds removed port");
        assertFalse(serversByPort.containsKey(victimPort), "serversByPort still holds removed port");

        // Discovery file no longer lists the removed engine.
        JsonNode root = readDiscoveryFile();
        assertEquals(List.of("127.0.0.1:" + (basePort - 1)),
                hostList(root, "mock.prefill.hosts.address"));
        assertEquals(List.of(), hostList(root, "mock.decode.hosts.address"));

        // /snapshot has no residue either.
        for (JsonNode engine : getJson("/snapshot").path("engines")) {
            assertFalse(("127.0.0.1:" + victimPort).equals(engine.path("grpc_addr").asText()),
                    "removed engine still visible in /snapshot");
        }
    }

    @Test
    void removeEngineReportsInflightCountersForBusyEngine() throws Exception {
        // 2000 ms decode steps keep the request in-flight across the remove call.
        int basePort = startCluster(model("10", 2000.0), 0, 1);
        int port = basePort;

        StreamCollector<EngineRpcService.GenerateOutputsPB> collector = new StreamCollector<>();
        services.get(port).generateStreamCall(input(6001, 10), collector);

        awaitCondition(() -> services.get(port).getRunningCount() >= 1, 2_000,
                "decode request should be running before remove");

        JsonNode removed = postOk("/remove_engine", "{\"port\":" + port + "}");
        assertTrue(removed.path("running_at_removal").asInt() >= 1,
                "running_at_removal should report the in-flight request, got "
                        + removed.path("running_at_removal"));
        awaitPortRefused(port);
    }

    @Test
    void addedEngineServesFullRequestPipeline() throws Exception {
        int basePort = startCluster(model("10", 1.0), 1, 1);

        // Prefill leg: dynamically added prefill engine accepts enqueueBatch and
        // drives the requests to finished via the completion scheduler.
        JsonNode addedPrefill = postOk("/add_engine", "{\"role\":\"prefill\"}");
        int prefillPort = addedPrefill.path("port").asInt();
        JavaMockEngineCluster.FastRpcService prefill = services.get(prefillPort);
        assertNotNull(prefill, "added prefill engine must be in the services map");

        EngineRpcService.EnqueueBatchResponsePB ack = enqueue(prefill,
                batch(31, slot(0, input(101, 100), input(102, 200))));
        assertEquals(2, ack.getSuccessesCount());

        EngineRpcService.WorkerStatusPB finished = awaitStatus(prefill,
                status -> status.getFinishedTaskListCount() == 2, 2_000);
        assertTrue(finished.getFinishedTaskListList().stream()
                .allMatch(task -> task.getExecutionTimeMs() == 10));

        // Decode leg: dynamically added decode engine runs generateStreamCall
        // end-to-end (decode queue scheduling → completion → streamed output).
        JsonNode addedDecode = postOk("/add_engine", "{\"role\":\"decode\"}");
        int decodePort = addedDecode.path("port").asInt();
        JavaMockEngineCluster.FastRpcService decode = services.get(decodePort);
        assertNotNull(decode, "added decode engine must be in the services map");

        StreamCollector<EngineRpcService.GenerateOutputsPB> collector = new StreamCollector<>();
        decode.generateStreamCall(input(201, 10), collector);
        assertTrue(collector.done.await(10, TimeUnit.SECONDS),
                "generateStreamCall on the added decode engine did not complete");
        assertNull(collector.error, "generateStreamCall failed: " + collector.error);
        assertEquals(1, collector.values.size(), "expected exactly one streamed output frame");
    }

    @Test
    void concurrentAddRemoveKeepsDiscoveryFileConsistentWithServices() throws Exception {
        startCluster(model("10", 1.0), 1, 1);

        // Seed a few dynamic engines first so the concurrent removes always have
        // non-bootstrap victims to pick from.
        for (int i = 0; i < 4; i++) {
            postOk("/add_engine", "{\"role\":\"" + (i % 2 == 0 ? "prefill" : "decode") + "\"}");
        }
        Set<Integer> bootstrapPorts = Set.copyOf(services.keySet());

        ExecutorService pool = Executors.newFixedThreadPool(8);
        List<Future<Integer>> outcomes = new ArrayList<>();
        try {
            for (int i = 0; i < 12; i++) {
                final boolean prefill = i % 2 == 0;
                outcomes.add(pool.submit(() -> post("/add_engine",
                        "{\"role\":\"" + (prefill ? "prefill" : "decode") + "\"}").statusCode()));
            }
            for (int i = 0; i < 12; i++) {
                outcomes.add(pool.submit(() -> {
                    List<Integer> candidates = services.keySet().stream()
                            .filter(p -> !bootstrapPorts.contains(p)).toList();
                    if (candidates.isEmpty()) {
                        return 0; // nothing to remove this round — fine
                    }
                    int victim = candidates.get(
                            ThreadLocalRandom.current().nextInt(candidates.size()));
                    return post("/remove_engine", "{\"port\":" + victim + "}").statusCode();
                }));
            }
            for (Future<Integer> outcome : outcomes) {
                // Every mutation either succeeded or cleanly failed with a
                // documented status (409 conflict / 404 victim gone); both leave
                // the file consistent. Anything else fails here.
                int code = outcome.get(60, TimeUnit.SECONDS);
                assertTrue(code == 200 || code == 404 || code == 409 || code == 500,
                        "unexpected status " + code + " from concurrent mutation");
            }
        } finally {
            pool.shutdownNow();
        }

        // The file must parse completely (no torn write window) and its entry
        // set must equal the live services map, role by role.
        JsonNode root = readDiscoveryFile();
        assertEquals(expectedAddresses("PREFILL"),
                hostList(root, "mock.prefill.hosts.address"),
                "discovery file diverged from services map (prefill)");
        assertEquals(expectedAddresses("DECODE"),
                hostList(root, "mock.decode.hosts.address"),
                "discovery file diverged from services map (decode)");
        // And no temporary files are left behind.
        try (var files = Files.list(tempDir)) {
            assertTrue(files.noneMatch(p -> p.getFileName().toString().endsWith(".tmp")),
                    "leftover .tmp discovery file after concurrent rewrites");
        }
    }

    @Test
    void addEngineRejectsBadRoleConflictingPortAndUnknownEngine() throws Exception {
        int basePort = startCluster(model("10", 1.0), 1, 1);

        assertEquals(400, post("/add_engine", "{\"role\":\"worker\"}").statusCode(),
                "unknown role must be rejected");
        assertEquals(400, post("/add_engine", "{}").statusCode(),
                "missing role must be rejected");
        assertEquals(409, post("/add_engine", "{\"role\":\"decode\",\"port\":" + basePort + "}")
                        .statusCode(),
                "port already in use must be rejected");
        assertEquals(404, post("/remove_engine", "{\"port\":59999}").statusCode(),
                "removing an unknown engine must 404");
    }

    // ════════════════════════════════════════════════════════════════
    //  Cluster bootstrap
    // ════════════════════════════════════════════════════════════════

    /**
     * Claim the next 32-port block with every port bindable right now.
     *
     * <p>The allocator restarts at its fixed base (63200) in every JVM, so
     * residue from a previous suite run (a not-yet-reaped socket from the
     * last JVM, or any other squatter on one of the ports) fails the
     * wildcard bind inside startEngine — the observed "Failed to bind
     * 0.0.0.0:63233" flake on back-to-back runs. Both the bootstrap engines
     * and the dynamically added ones (auto-allocated as max+1) take
     * consecutive ports inside the block, so the whole block must be free;
     * occupied blocks are skipped. The probe binds exactly like the engine
     * does (wildcard address, same JVM-default socket options), so anything
     * that would fail the engine bind also fails the probe. Mirrors the
     * established {@code ComprehensiveFaultInjectionTest.allocatePortBlock}
     * pattern.
     */
    private static int allocatePortBlock() {
        for (int attempt = 0; attempt < 20; attempt++) {
            int basePort = PORT_ALLOCATOR.getAndAdd(32);
            boolean allFree = true;
            for (int port = basePort; port < basePort + 32; port++) {
                // Wildcard bind with the same JVM-default socket options as
                // the NettyServerBuilder.forPort() bind inside startEngine,
                // so probe success/failure predicts the engine bind 1:1.
                try (ServerSocket probe = new ServerSocket()) {
                    probe.bind(new InetSocketAddress(port), 1);
                } catch (IOException e) {
                    allFree = false;
                    break;
                }
            }
            if (allFree) {
                return basePort;
            }
        }
        throw new IllegalStateException("no bindable 32-port block after 20 attempts");
    }

    /**
     * Boot a real cluster (gRPC servers + control server + discovery file +
     * dynamic engine manager), mirroring {@code JavaMockEngineCluster.main}.
     */
    private int startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        int basePort = allocatePortBlock();
        scheduler = Executors.newScheduledThreadPool(4, r -> {
            Thread thread = new Thread(r, "scale-test-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        bossGroup = new NioEventLoopGroup(1);
        workerGroup = new NioEventLoopGroup(4);
        services = new ConcurrentHashMap<>();
        serversByPort = new ConcurrentHashMap<>();
        JavaMockEngineCluster.ClusterStats stats = new JavaMockEngineCluster.ClusterStats();
        JavaMockEngineCluster.Config config = new JavaMockEngineCluster.Config();
        config.host = "127.0.0.1";
        // This suite pins discovery-file contents with literal 127.0.0.1
        // addresses; disable the unique-IP advertisement so the assertions
        // below keep their original (upstream) semantics. Unique-IP dynamic
        // add behavior is covered by UniqueEngineIpsTest.
        config.uniqueEngineIps = false;

        for (int i = 0; i < nPrefill; i++) {
            JavaMockEngineCluster.startEngine(config, model, serversByPort, bossGroup,
                    workerGroup, services, scheduler, stats, "prefill", "prefill-" + i,
                    basePort + i, i);
        }
        for (int i = 0; i < nDecode; i++) {
            JavaMockEngineCluster.startEngine(config, model, serversByPort, bossGroup,
                    workerGroup, services, scheduler, stats, "decode", "decode-" + i,
                    basePort + nPrefill + i, nPrefill + i);
        }

        discoveryFile = tempDir.resolve("discovery-" + System.nanoTime() + ".json");
        discoveryFileStore = new DiscoveryFileStore(discoveryFile.toString(),
                config.prefillDomain, config.decodeDomain);
        discoveryFileStore.rewrite(services);
        engineManager = new DynamicEngineManager(config, model, services, serversByPort,
                bossGroup, workerGroup, scheduler, stats, discoveryFileStore, null);
        controlServer = new MockControlServer(services, serversByPort, bossGroup, workerGroup,
                "127.0.0.1", 0, engineManager);
        controlServer.start();
        return basePort;
    }

    // ════════════════════════════════════════════════════════════════
    //  HTTP helpers
    // ════════════════════════════════════════════════════════════════

    private JsonNode postOk(String path, String body) throws Exception {
        HttpResponse<String> response = post(path, body);
        assertEquals(200, response.statusCode(), "POST " + path + " -> " + response.body());
        return MAPPER.readTree(response.body());
    }

    private HttpResponse<String> post(String path, String body) throws Exception {
        return HTTP_CLIENT.send(HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:" + controlServer.getPort() + path))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build(), HttpResponse.BodyHandlers.ofString());
    }

    private JsonNode getJson(String path) throws Exception {
        HttpResponse<String> response = HTTP_CLIENT.send(HttpRequest.newBuilder()
                .uri(URI.create("http://127.0.0.1:" + controlServer.getPort() + path))
                .GET().build(), HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), "GET " + path + " failed");
        return MAPPER.readTree(response.body());
    }

    // ════════════════════════════════════════════════════════════════
    //  gRPC / discovery-file helpers
    // ════════════════════════════════════════════════════════════════

    /** Real gRPC round-trip against the engine on {@code port}. */
    private EngineRpcService.WorkerStatusPB grpcWorkerStatus(int port) {
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", port)
                .usePlaintext()
                .build();
        try {
            return RpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(5, TimeUnit.SECONDS)
                    .getWorkerStatus(EngineRpcService.StatusVersionPB.newBuilder()
                            .setLatestFinishedVersion(0)
                            .build());
        } finally {
            channel.shutdownNow();
        }
    }

    /** Wait until the port refuses TCP connections (listener closed by remove). */
    private static void awaitPortRefused(int port) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            try (Socket socket = new Socket()) {
                socket.connect(new InetSocketAddress("127.0.0.1", port), 200);
            } catch (IOException e) {
                return; // connection refused — listener is gone
            }
            Thread.sleep(20);
        }
        fail("port " + port + " still accepts connections after remove");
    }

    private JsonNode readDiscoveryFile() throws IOException {
        return MAPPER.readTree(Files.readString(discoveryFile));
    }

    private static List<String> hostList(JsonNode root, String domain) {
        List<String> hosts = new ArrayList<>();
        for (JsonNode node : root.path(domain)) {
            hosts.add(node.asText());
        }
        return hosts;
    }

    /** All live engines of {@code role} as ip:httpPort strings ordered by gRPC port. */
    private List<String> expectedAddresses(String role) {
        return services.values().stream()
                .filter(service -> role.equals(service.getRoleName()))
                .sorted(Comparator.comparingInt(JavaMockEngineCluster.FastRpcService::getGrpcPort))
                .map(service -> service.getHost() + ":" + (service.getGrpcPort() - 1))
                .toList();
    }

    private static void awaitCondition(BooleanSupplier condition, long timeoutMs, String message)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(10);
        }
        fail(message);
    }

    private interface BooleanSupplier {
        boolean getAsBoolean();
    }

    // ════════════════════════════════════════════════════════════════
    //  Model / request helpers (JavaMockEngineClusterTest style)
    // ════════════════════════════════════════════════════════════════

    private MockPerformanceModel model(String formula, double decodeStepMs) throws Exception {
        Path performance = tempDir.resolve("performance-" + System.nanoTime() + ".json");
        Path master = tempDir.resolve("master-" + System.nanoTime() + ".json");
        MAPPER.writeValue(performance.toFile(), Map.of(
                "block_size", 1024,
                "sleep_scale", 1.0,
                "prefill", Map.of("scale", 1.0),
                "decode", Map.of("scale", 1.0,
                        "step_ms_by_batch", List.of(List.of(1, decodeStepMs)))));
        // Local intake: MockPerformanceModel reads the prefill expression from the
        // FLEXLB_CONFIG env (FormulaEstimatorConfig), not the upstream
        // PREFILL_TIME_FORMULA env — use the shared test helper.
        MockMasterConfig.writeWithPrefillExpression(master, formula);
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

    private static EngineRpcService.WorkerStatusPB awaitStatus(
            JavaMockEngineCluster.FastRpcService service,
            Predicate<EngineRpcService.WorkerStatusPB> predicate,
            long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        EngineRpcService.WorkerStatusPB last = null;
        while (System.nanoTime() < deadline) {
            last = status(service);
            if (predicate.test(last)) {
                return last;
            }
            Thread.sleep(5);
        }
        fail("status condition not reached, last status=" + last);
        return last;
    }

    private static <T> T unary(Consumer<StreamObserver<T>> invocation) {
        AtomicReference<T> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        invocation.accept(new StreamObserver<>() {
            @Override
            public void onNext(T value) {
                response.set(value);
            }

            @Override
            public void onError(Throwable throwable) {
                error.set(throwable);
            }

            @Override
            public void onCompleted() {
            }
        });
        if (error.get() != null) {
            throw new AssertionError(error.get());
        }
        assertNotNull(response.get(), "unary response");
        return response.get();
    }

    /** Async collector for streaming RPCs (generateStreamCall). */
    private static final class StreamCollector<T> implements StreamObserver<T> {
        final List<T> values = new CopyOnWriteArrayList<>();
        final CountDownLatch done = new CountDownLatch(1);
        volatile Throwable error;

        @Override
        public void onNext(T value) {
            values.add(value);
        }

        @Override
        public void onError(Throwable throwable) {
            error = throwable;
            done.countDown();
        }

        @Override
        public void onCompleted() {
            done.countDown();
        }
    }
}
