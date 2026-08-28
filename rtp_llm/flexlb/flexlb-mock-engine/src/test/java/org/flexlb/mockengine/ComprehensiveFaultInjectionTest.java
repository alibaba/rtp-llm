package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.ServerSocket;
import java.net.http.HttpClient;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.input;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.flexlb.mockengine.MockEngineTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Comprehensive fault injection test covering all fault injection types
 * supported by the Java mock engine's FaultInjectionConfig.
 *
 * <p>Each test method exercises one fault injection type via the HTTP control
 * API ({@link MockControlServer}) and/or direct gRPC calls
 * ({@link JavaMockEngineCluster.FastRpcService}), verifying the fault
 * takes effect and that clearing it restores normal behaviour.
 *
 * <p>API notes:
 * <ul>
 *   <li>The HTTP /inject endpoint accepts POST with JSON body
 *       {@code {"port":N,"type":"...","enabled":true,...}}, not GET with
 *       query parameters.</li>
 *   <li>Supported inject types: enqueue_error, generate_error, fetch_error,
 *       no_respond, kv_pressure, queue_depth, crash_after, enqueue_delay,
 *       generate_delay.</li>
 *   <li>/start_engine automatically clears fault injection config and resets
 *       enqueueCount, so crash recovery no longer requires /clear_inject
 *       first.</li>
 * </ul>
 */
class ComprehensiveFaultInjectionTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final AtomicInteger PORT_ALLOCATOR = new AtomicInteger(62900);
    private static final long TOTAL_KV_TOKENS = 6_291_456L;

    @TempDir
    Path tempDir;

    private ScheduledExecutorService scheduler;
    private EventLoopGroup bossGroup;
    private EventLoopGroup workerGroup;
    private MockControlServer controlServer;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private Map<Integer, Server> serversByPort;
    private List<JavaMockEngineCluster.FastRpcService> prefillServices;
    private List<JavaMockEngineCluster.FastRpcService> decodeServices;

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
        prefillServices = null;
        decodeServices = null;
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 1: enqueue_error
    // ════════════════════════════════════════════════════════════════

    @Test
    void enqueueErrorReturnsErrorsThenRecovers() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 2);
        int prefillPort = basePort;
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        try {
            // Inject enqueue_error via HTTP
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort + ",\"type\":\"enqueue_error\",\"enabled\":true}");
            assertTrue(prefill.getFaultConfig().isFailOnEnqueue());

            // Enqueue 5 requests — all should get errors
            for (int i = 1; i <= 5; i++) {
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(1000 + i, slot(0, input(i, 10))));
                assertEquals(0, response.getSuccessesCount(),
                        "request " + i + " should have 0 successes under enqueue_error");
                assertEquals(1, response.getErrorsCount(),
                        "request " + i + " should have 1 error under enqueue_error");
            }

            // Clear injection
            httpPost(controlServer.getPort(), "/clear_inject",
                    "{\"port\":" + prefillPort + "}");
            assertFalse(prefill.getFaultConfig().isFailOnEnqueue());

            // Enqueue 5 more — all should succeed
            for (int i = 6; i <= 10; i++) {
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(1000 + i, slot(0, input(i, 10))));
                assertEquals(1, response.getSuccessesCount(),
                        "request " + i + " should succeed after clearing enqueue_error");
                assertEquals(0, response.getErrorsCount(),
                        "request " + i + " should have 0 errors after clearing");
            }

            // Verify no leak
            awaitAllInflightZero(5_000);
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertFalse(service.isLeakDetected(),
                        "no leak on port " + service.getGrpcPort());
            }
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 2: generate_error
    // ════════════════════════════════════════════════════════════════

    @Test
    void generateErrorReturnsErrorThenRecovers() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 2);
        int prefillPort = basePort;

        try {
            // Inject generate_error via HTTP
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort + ",\"type\":\"generate_error\",\"enabled\":true}");
            assertTrue(prefillServices.get(0).getFaultConfig().isGenerateError());

            // Call generateStreamCall 5 times — all should return error
            for (int i = 1; i <= 5; i++) {
                GenerateResult result = generateStream(
                        prefillServices.get(0), input(i, 10), 3_000);
                assertTrue(result.completed(), "generateStreamCall " + i + " should complete (onError)");
                assertNotNull(result.error(), "generateStreamCall " + i + " should have error");
                assertTrue(result.error().getMessage().contains("generate_error"),
                        "error message should contain 'generate_error', got: " + result.error().getMessage());
            }

            // Clear injection
            httpPost(controlServer.getPort(), "/clear_inject",
                    "{\"port\":" + prefillPort + "}");
            assertFalse(prefillServices.get(0).getFaultConfig().isGenerateError());

            // Call generateStreamCall 5 more times — all should succeed
            for (int i = 6; i <= 10; i++) {
                GenerateResult result = generateStream(
                        prefillServices.get(0), input(i, 10), 5_000);
                assertTrue(result.completed(), "generateStreamCall " + i + " should complete after clear");
                assertNotNull(result.response(), "generateStreamCall " + i + " should have response");
                assertEquals(i, result.response().getRequestId());
            }

            // Verify no leak
            awaitAllInflightZero(5_000);
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertFalse(service.isLeakDetected(),
                        "no leak on port " + service.getGrpcPort());
            }
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 3: no_respond
    // ════════════════════════════════════════════════════════════════

    @Test
    void noRespondTimesOutThenRecovers() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 2);
        int prefillPort = basePort;

        try {
            // Inject no_respond via HTTP
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort + ",\"type\":\"no_respond\",\"enabled\":true}");
            assertTrue(prefillServices.get(0).getFaultConfig().isNoRespond());

            // Call generateStreamCall 3 times — all should time out (no response within 2s)
            for (int i = 1; i <= 3; i++) {
                long start = System.nanoTime();
                GenerateResult result = generateStream(
                        prefillServices.get(0), input(i, 10), 2_000);
                long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
                assertFalse(result.completed(),
                        "generateStreamCall " + i + " should time out under no_respond");
                assertTrue(elapsedMs >= 1_500,
                        "generateStreamCall " + i + " should wait ~2s before timeout, got " + elapsedMs + "ms");
            }

            // Clear injection
            httpPost(controlServer.getPort(), "/clear_inject",
                    "{\"port\":" + prefillPort + "}");
            assertFalse(prefillServices.get(0).getFaultConfig().isNoRespond());

            // Call generateStreamCall 3 more times — all should succeed
            for (int i = 4; i <= 6; i++) {
                GenerateResult result = generateStream(
                        prefillServices.get(0), input(i, 10), 5_000);
                assertTrue(result.completed(), "generateStreamCall " + i + " should complete after clear");
                assertNotNull(result.response(), "generateStreamCall " + i + " should have response");
            }

            // Verify no leak
            awaitAllInflightZero(5_000);
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertFalse(service.isLeakDetected(),
                        "no leak on port " + service.getGrpcPort());
            }
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 4: kv_pressure
    // ════════════════════════════════════════════════════════════════

    @Test
    void kvPressureInflatesKvUsage() throws Exception {
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 2);
        int prefillPort = basePort;
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        try {
            // Verify baseline: available == total
            EngineRpcService.WorkerStatusPB before = workerStatus(prefill, 0);
            assertEquals(TOTAL_KV_TOKENS, before.getAvailableKvCache(),
                    "baseline available KV should equal total");
            double beforeRatio = 1.0 - (double) before.getAvailableKvCache() / before.getTotalKvCache();
            assertEquals(0.0, beforeRatio, 0.001, "baseline KV ratio should be ~0");

            // Set KV pressure via HTTP — using 4M tokens for >0.5 ratio
            // (50000 as suggested in the task spec is insufficient for >0.5 with total=6,291,456)
            long pressureTokens = 4_000_000L;
            httpPost(controlServer.getPort(), "/set_kv_pressure",
                    "{\"port\":" + prefillPort + ",\"tokens\":" + pressureTokens + "}");
            assertEquals(pressureTokens, prefill.getFaultConfig().getKvPressureTokens());

            // Get worker status — verify pressure
            EngineRpcService.WorkerStatusPB after = workerStatus(prefill, 0);
            long usedKv = after.getTotalKvCache() - after.getAvailableKvCache();
            double ratio = (double) usedKv / after.getTotalKvCache();
            assertTrue(ratio > 0.5,
                    "KV pressure should make used ratio > 0.5, got " + ratio
                            + " (used=" + usedKv + ", total=" + after.getTotalKvCache() + ")");
            assertTrue(after.getAvailableKvCache() < after.getTotalKvCache(),
                    "available KV should be less than total under pressure");

            // Clear KV pressure
            httpPost(controlServer.getPort(), "/set_kv_pressure",
                    "{\"port\":" + prefillPort + ",\"tokens\":0}");
            assertEquals(0, prefill.getFaultConfig().getKvPressureTokens());

            // Verify recovery
            EngineRpcService.WorkerStatusPB recovered = workerStatus(prefill, 0);
            assertEquals(TOTAL_KV_TOKENS, recovered.getAvailableKvCache(),
                    "available KV should return to total after clearing pressure");
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 5: queue_depth_limit
    // ════════════════════════════════════════════════════════════════

    @Test
    void queueDepthLimitRejectsExcessRequests() throws Exception {
        // Use a long prefill formula so requests stay in-flight
        MockPerformanceModel model = model("500");
        int basePort = startCluster(model, 1, 2);
        int prefillPort = basePort;
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        try {
            // Set queue depth limit to 2 via HTTP
            httpPost(controlServer.getPort(), "/set_queue_depth",
                    "{\"port\":" + prefillPort + ",\"depth\":2}");
            assertEquals(2, prefill.getFaultConfig().getQueueDepthLimit());

            // Enqueue 5 requests one at a time
            int accepted = 0;
            int rejected = 0;
            for (int i = 1; i <= 5; i++) {
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(2000 + i, slot(0, input(i, 10))));
                if (response.getSuccessesCount() > 0) {
                    accepted++;
                } else if (response.getErrorsCount() > 0) {
                    rejected++;
                }
            }

            // Verify: at most 2 accepted, 3 rejected
            assertTrue(accepted <= 2,
                    "at most 2 requests should be accepted, got " + accepted);
            assertEquals(3, rejected,
                    "3 requests should be rejected by queue depth, got " + rejected);

            // Clear queue depth limit
            httpPost(controlServer.getPort(), "/set_queue_depth",
                    "{\"port\":" + prefillPort + ",\"depth\":0}");
            assertEquals(0, prefill.getFaultConfig().getQueueDepthLimit());

            // Wait for in-flight requests to drain
            awaitAllInflightZero(10_000);

            // Enqueue after clearing — should succeed
            EngineRpcService.EnqueueBatchResponsePB response =
                    enqueue(prefill, batch(2999, slot(0, input(99, 10))));
            assertEquals(1, response.getSuccessesCount(),
                    "enqueue should succeed after clearing queue depth limit");

            awaitAllInflightZero(10_000);
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertFalse(service.isLeakDetected(),
                        "no leak on port " + service.getGrpcPort());
            }
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 6: crash_after_n
    // ════════════════════════════════════════════════════════════════

    @Test
    void crashAfterNStopsEngineThenRecovers() throws Exception {
        MockPerformanceModel model = model("10");
        // Need real gRPC servers for /start_engine to work
        int basePort = startClusterWithGrpc(model, 1, 2);
        int prefillPort = basePort;
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        try {
            // Inject crash_after with n=3 via HTTP
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort + ",\"type\":\"crash_after\",\"enabled\":true,\"n\":3}");
            assertEquals(3, prefill.getFaultConfig().getCrashAfterNRequests());

            // Enqueue requests one at a time
            int succeeded = 0;
            int emptyAfter = 0;
            for (int i = 1; i <= 10; i++) {
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(3000 + i, slot(0, input(i, 10))));
                if (response.getSuccessesCount() > 0) {
                    succeeded++;
                } else {
                    // After crash, response has 0 successes and 0 errors
                    emptyAfter++;
                }
                if (i == 3) {
                    assertTrue(prefill.isStopped(),
                            "engine should be stopped after 3rd request (crash_after n=3)");
                }
            }

            // First 2 succeed, 3rd triggers crash (empty response), 4-10 are stopped (empty)
            assertEquals(2, succeeded,
                    "2 requests should succeed before crash, got " + succeeded);
            assertEquals(8, emptyAfter,
                    "8 requests should get empty response after crash, got " + emptyAfter);
            assertTrue(prefill.isStopped(), "engine should be stopped after crash");

            // Restart via /start_engine — this now auto-clears fault config
            // and resets enqueueCount, so no /clear_inject needed
            httpPost(controlServer.getPort(), "/start_engine",
                    "{\"port\":" + prefillPort + "}");
            assertFalse(prefill.isStopped(), "engine should be running after /start_engine");
            assertEquals(0, prefill.getFaultConfig().getCrashAfterNRequests(),
                    "start_engine should have cleared crash_after fault config");

            // Verify recovery — enqueue should succeed
            EngineRpcService.EnqueueBatchResponsePB response =
                    enqueue(prefill, batch(3999, slot(0, input(99, 10))));
            assertEquals(1, response.getSuccessesCount(),
                    "enqueue should succeed after engine restart");

            awaitAllInflightZero(5_000);
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertFalse(service.isLeakDetected(),
                        "no leak on port " + service.getGrpcPort());
            }
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 7: enqueue_delay
    // ════════════════════════════════════════════════════════════════

    @Test
    void enqueueDelayAddsLatency() throws Exception {
        // enqueue_delay is now supported via HTTP /inject with delay_ms parameter.
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 2);
        int prefillPort = basePort;
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        try {
            // Set enqueue delay via HTTP /inject
            long delayMs = 100;
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort + ",\"type\":\"enqueue_delay\",\"enabled\":true,\"delay_ms\":" + delayMs + "}");
            assertEquals(delayMs, prefill.getFaultConfig().getEnqueueDelayMs());

            // Enqueue 5 requests — each should take at least delayMs
            for (int i = 1; i <= 5; i++) {
                long start = System.nanoTime();
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(4000 + i, slot(0, input(i, 10))));
                long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
                assertEquals(1, response.getSuccessesCount(),
                        "request " + i + " should succeed despite delay");
                assertTrue(elapsedMs >= delayMs,
                        "request " + i + " should take >= " + delayMs + "ms, got " + elapsedMs + "ms");
            }

            // Clear injection via HTTP /clear_inject
            httpPost(controlServer.getPort(), "/clear_inject",
                    "{\"port\":" + prefillPort + "}");
            assertEquals(0, prefill.getFaultConfig().getEnqueueDelayMs());

            // Enqueue 5 more — should be fast (no delay)
            for (int i = 6; i <= 10; i++) {
                long start = System.nanoTime();
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(4000 + i, slot(0, input(i, 10))));
                long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
                assertEquals(1, response.getSuccessesCount(),
                        "request " + i + " should succeed after clearing delay");
                assertTrue(elapsedMs < delayMs,
                        "request " + i + " should be fast after clearing delay, got " + elapsedMs + "ms");
            }

            awaitAllInflightZero(5_000);
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertFalse(service.isLeakDetected(),
                        "no leak on port " + service.getGrpcPort());
            }
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Test 8: generate_delay (via /set_perf prefill_ms override)
    // ════════════════════════════════════════════════════════════════

    @Test
    void generateDelayAddsLatency() throws Exception {
        // generate_delay is now supported via HTTP /inject with delay_ms parameter.
        // It delays the prefill completion (generate) by the specified ms,
        // independent of the performance model's prefill_ms formula.
        MockPerformanceModel model = model("10");
        int basePort = startCluster(model, 1, 2);
        int prefillPort = basePort;
        JavaMockEngineCluster.FastRpcService prefill = prefillServices.get(0);

        try {
            // Set generate delay via HTTP /inject
            long delayMs = 200;
            httpPost(controlServer.getPort(), "/inject",
                    "{\"port\":" + prefillPort + ",\"type\":\"generate_delay\",\"enabled\":true,\"delay_ms\":" + delayMs + "}");
            assertEquals(delayMs, prefill.getFaultConfig().getGenerateDelayMs());

            // Call generateStreamCall 5 times — each should take at least delayMs
            for (int i = 1; i <= 5; i++) {
                long start = System.nanoTime();
                GenerateResult result = generateStream(
                        prefill, input(i, 10), 5_000);
                long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
                assertTrue(result.completed(),
                        "generateStreamCall " + i + " should complete");
                assertNotNull(result.response(),
                        "generateStreamCall " + i + " should have response");
                assertTrue(elapsedMs >= delayMs * 0.8,
                        "generateStreamCall " + i + " should take >= ~" + delayMs
                                + "ms, got " + elapsedMs + "ms");
            }

            // Clear generate delay via HTTP /clear_inject
            httpPost(controlServer.getPort(), "/clear_inject",
                    "{\"port\":" + prefillPort + "}");
            assertEquals(0, prefill.getFaultConfig().getGenerateDelayMs());

            // Call generateStreamCall 5 more times — should be fast
            for (int i = 6; i <= 10; i++) {
                long start = System.nanoTime();
                GenerateResult result = generateStream(
                        prefill, input(i, 10), 5_000);
                long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start);
                assertTrue(result.completed(),
                        "generateStreamCall " + i + " should complete after clearing");
                assertNotNull(result.response(),
                        "generateStreamCall " + i + " should have response");
                assertTrue(elapsedMs < delayMs,
                        "generateStreamCall " + i + " should be fast after clearing, got " + elapsedMs + "ms");
            }

            awaitAllInflightZero(5_000);
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                assertFalse(service.isLeakDetected(),
                        "no leak on port " + service.getGrpcPort());
            }
        } finally {
            cleanupCluster();
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  Cluster setup helpers
    // ════════════════════════════════════════════════════════════════

    /**
     * Claim the next 10-port block whose first {@code needed} ports are all
     * bindable right now. The allocator's ports sit inside the OS ephemeral
     * range, so a stray outbound connection from an unrelated process (e.g. a
     * lingering CLOSE_WAIT socket) can squat on a deterministic port and fail
     * every full-suite run; skipping occupied blocks keeps the tests hermetic.
     */
    private static int allocatePortBlock(int needed) {
        for (int attempt = 0; attempt < 20; attempt++) {
            int basePort = PORT_ALLOCATOR.getAndAdd(10);
            boolean allFree = true;
            for (int i = 0; i < needed; i++) {
                try (ServerSocket probe = new ServerSocket(basePort + i)) {
                    probe.setReuseAddress(true);
                } catch (IOException e) {
                    allFree = false;
                    break;
                }
            }
            if (allFree) {
                return basePort;
            }
        }
        throw new IllegalStateException("no bindable 10-port block after 20 attempts");
    }

    /**
     * Start a cluster with direct service calls (no real gRPC servers).
     * Suitable for tests that use /inject, /clear_inject, /set_kv_pressure,
     * /set_queue_depth, /set_perf but NOT /start_engine.
     */
    private int startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        int basePort = allocatePortBlock(nPrefill + nDecode);
        scheduler = Executors.newScheduledThreadPool(8, r -> {
            Thread t = new Thread(r, "mock-engine-scheduler");
            t.setDaemon(true);
            return t;
        });
        services = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

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
        return basePort;
    }

    /**
     * Start a cluster with real gRPC servers. Required for tests that
     * use /start_engine (which needs EventLoopGroups to rebuild servers).
     */
    private int startClusterWithGrpc(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        int basePort = allocatePortBlock(nPrefill + nDecode);
        scheduler = Executors.newScheduledThreadPool(8, r -> {
            Thread t = new Thread(r, "mock-engine-scheduler");
            t.setDaemon(true);
            return t;
        });
        bossGroup = new NioEventLoopGroup(1);
        workerGroup = new NioEventLoopGroup(8);
        services = new ConcurrentHashMap<>();
        serversByPort = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

        for (int i = 0; i < nPrefill; i++) {
            int port = basePort + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "prefill", EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            prefillServices.add(service);
            Server server = NettyServerBuilder.forPort(port)
                    .bossEventLoopGroup(bossGroup)
                    .workerEventLoopGroup(workerGroup)
                    .channelType(NioServerSocketChannel.class)
                    .directExecutor()
                    .maxInboundMessageSize(16 * 1024 * 1024)
                    .addService(service)
                    .build()
                    .start();
            serversByPort.put(port, server);
        }
        for (int i = 0; i < nDecode; i++) {
            int port = basePort + nPrefill + i;
            JavaMockEngineCluster.FastRpcService service = new JavaMockEngineCluster.FastRpcService(
                    "decode", EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE,
                    port, services, scheduler, model, 100,
                    new JavaMockEngineCluster.ClusterStats());
            services.put(port, service);
            decodeServices.add(service);
            Server server = NettyServerBuilder.forPort(port)
                    .bossEventLoopGroup(bossGroup)
                    .workerEventLoopGroup(workerGroup)
                    .channelType(NioServerSocketChannel.class)
                    .directExecutor()
                    .maxInboundMessageSize(16 * 1024 * 1024)
                    .addService(service)
                    .build()
                    .start();
            serversByPort.put(port, server);
        }

        controlServer = new MockControlServer(services, serversByPort, bossGroup, workerGroup, "127.0.0.1", 0);
        controlServer.start();
        return basePort;
    }

    /**
     * Mid-test cleanup (for finally blocks). The @AfterEach also cleans up
     * as a safety net.
     */
    private void cleanupCluster() throws InterruptedException {
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
            bossGroup.shutdownGracefully(0, 1, TimeUnit.SECONDS);
            bossGroup = null;
        }
        if (workerGroup != null) {
            workerGroup.shutdownGracefully(0, 1, TimeUnit.SECONDS);
            workerGroup = null;
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(2, TimeUnit.SECONDS);
            scheduler = null;
        }
        prefillServices = null;
        decodeServices = null;
    }

    // ════════════════════════════════════════════════════════════════
    //  Polling helpers
    // ════════════════════════════════════════════════════════════════

    private void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        if (services == null) {
            return;
        }
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (services.values().stream().allMatch(s -> s.getInflightCount() == 0)) {
                return;
            }
            Thread.sleep(10);
        }
        StringBuilder sb = new StringBuilder("inflight not zero: ");
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            sb.append("port=").append(service.getGrpcPort())
                    .append(" inflight=").append(service.getInflightCount()).append(" ");
        }
        fail(sb.toString());
    }

    // ════════════════════════════════════════════════════════════════
    //  HTTP helpers
    // ════════════════════════════════════════════════════════════════

    private static String httpGet(int port, String path) throws Exception {
        return MockEngineTestSupport.httpGet(port, path);
    }

    private static String httpPost(int port, String path, String body) throws Exception {
        return MockEngineTestSupport.httpPost(port, path, body);
    }

    // ════════════════════════════════════════════════════════════════
    //  Model helper
    // ════════════════════════════════════════════════════════════════

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }

    /**
     * Call generateStreamCall and return the result with a configurable timeout.
     * Used for generate_error (expects error), no_respond (expects timeout),
     * and normal generate (expects response).
     */
    private static GenerateResult generateStream(
            JavaMockEngineCluster.FastRpcService service,
            EngineRpcService.GenerateInputPB request,
            long timeoutMs) throws InterruptedException {
        AtomicReference<EngineRpcService.GenerateOutputsPB> response = new AtomicReference<>();
        AtomicReference<Throwable> error = new AtomicReference<>();
        CountDownLatch latch = new CountDownLatch(1);
        service.generateStreamCall(request, new StreamObserver<>() {
            @Override
            public void onNext(EngineRpcService.GenerateOutputsPB value) {
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
        boolean completed = latch.await(timeoutMs, TimeUnit.MILLISECONDS);
        return new GenerateResult(completed, response.get(), error.get());
    }

    // ════════════════════════════════════════════════════════════════
    //  Result record for generateStreamCall
    // ════════════════════════════════════════════════════════════════

    record GenerateResult(boolean completed,
                          EngineRpcService.GenerateOutputsPB response,
                          Throwable error) {
    }
}
