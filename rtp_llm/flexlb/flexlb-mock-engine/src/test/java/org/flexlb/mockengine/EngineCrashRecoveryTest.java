package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.net.http.HttpClient;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Engine crash/restart recovery test for the Java mock engine.
 *
 * <p>Starts a cluster with 2 prefill + 4 decode engines (real gRPC servers),
 * enqueues 20 requests, stops one decode engine mid-flight via the HTTP
 * /stop_engine endpoint, waits for remaining requests to drain, then restarts
 * the stopped engine via /start_engine and verifies health. Sends 10 more
 * requests post-restart and confirms they complete with no leak.
 */
class EngineCrashRecoveryTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 62400;

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
        }
        if (services != null) {
            for (JavaMockEngineCluster.FastRpcService service : services.values()) {
                service.shutdown();
            }
        }
        if (bossGroup != null) {
            bossGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        }
        if (workerGroup != null) {
            workerGroup.shutdownGracefully(0, 2, TimeUnit.SECONDS);
        }
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    @Test
    void engineCrashAndRecoveryNoLeak() throws Exception {
        MockPerformanceModel model = model("10");
        startCluster(model, 2, 4);

        // ── Phase 1: Enqueue 5 requests, then stop one decode engine ──
        enqueueBatch(prefillServices.get(0), 8000, 1, 5, decodeServices);

        int stoppedDecodePort = decodeServices.get(0).getGrpcPort();
        httpPost(controlServer.getPort(), "/stop_engine",
                "{\"port\":" + stoppedDecodePort + "}");
        assertTrue(decodeServices.get(0).isStopped(),
                "decode engine should be stopped after /stop_engine");

        // ── Phase 2: Enqueue remaining 15 requests (routing may hit stopped engine) ──
        enqueueBatch(prefillServices.get(1), 8001, 6, 15, decodeServices);

        // Wait for all inflight to drain on all engines
        awaitAllInflightZero(10_000);
        assertAllInflightZero();

        // Verify no leak detected on any engine (including the stopped one)
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected on port " + service.getGrpcPort());
        }

        // ── Phase 3: Restart the stopped engine ──
        httpPost(controlServer.getPort(), "/start_engine",
                "{\"port\":" + stoppedDecodePort + "}");
        assertFalse(decodeServices.get(0).isStopped(),
                "decode engine should be running after /start_engine");

        // Verify health endpoint reports all engines healthy
        String healthBody = httpGet(controlServer.getPort(), "/health");
        JsonNode healthJson = MAPPER.readTree(healthBody);
        assertTrue(healthJson.get("healthy").asBoolean(),
                "all engines should be healthy after restart");
        assertEquals(services.size(), healthJson.get("engines").asInt(),
                "engine count should match");

        // ── Phase 4: Send 10 more requests and verify they complete ──
        enqueueBatch(prefillServices.get(0), 8002, 21, 10, decodeServices);

        awaitAllInflightZero(10_000);
        assertAllInflightZero();

        // Final leak check on all engines
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected after recovery on port "
                            + service.getGrpcPort());
        }

        // Verify HTTP snapshot consistency
        String body = httpGet(controlServer.getPort(), "/snapshot");
        JsonNode snapshot = MAPPER.readTree(body).path("engines");
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port "
                            + engine.get("port").asInt());
            assertFalse(engine.get("leak_detected").asBoolean(),
                    "snapshot leak_detected should be false for port "
                            + engine.get("port").asInt());
        }
    }

    // ──────────── Cluster setup (with real gRPC servers) ────────────

    private void startCluster(MockPerformanceModel model, int nPrefill, int nDecode)
            throws IOException {
        scheduler = Executors.newScheduledThreadPool(8, runnable -> {
            Thread thread = new Thread(runnable, "mock-engine-scheduler");
            thread.setDaemon(true);
            return thread;
        });
        bossGroup = new NioEventLoopGroup(1);
        workerGroup = new NioEventLoopGroup(8);
        services = new ConcurrentHashMap<>();
        serversByPort = new ConcurrentHashMap<>();
        prefillServices = new ArrayList<>();
        decodeServices = new ArrayList<>();

        for (int i = 0; i < nPrefill; i++) {
            int port = BASE_PORT + i;
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
            int port = BASE_PORT + nPrefill + i;
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

        // HTTP control server with real EventLoopGroups and serversByPort
        // so /start_engine can rebuild the gRPC server
        controlServer = new MockControlServer(services, serversByPort, bossGroup, workerGroup, "127.0.0.1", 0);
        controlServer.start();
    }

    // ──────────── Polling helpers ────────────

    private void awaitAllInflightZero(long timeoutMs) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (System.nanoTime() < deadline) {
            if (services.values().stream()
                    .allMatch(s -> s.getInflightCount() == 0)) {
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

    private void assertAllInflightZero() {
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertEquals(0, service.getInflightCount(),
                    "inflight should be 0 for engine on port " + service.getGrpcPort());
        }
    }

    // ──────────── Batch enqueue helper ────────────

    private void enqueueBatch(JavaMockEngineCluster.FastRpcService prefill,
                              long batchId, int startRequestId, int count,
                              List<JavaMockEngineCluster.FastRpcService> decodeEngines) {
        EngineRpcService.GenerateInputPB[] inputs = new EngineRpcService.GenerateInputPB[count];
        for (int i = 0; i < count; i++) {
            int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
            inputs[i] = inputWithDecode(startRequestId + i, 10, decodePort);
        }
        enqueue(prefill, batch(batchId, slot(0, inputs)));
    }

    // ──────────── HTTP helpers ────────────

    private static String httpGet(int port, String path) throws Exception {
        return MockEngineTestSupport.httpGet(port, path);
    }

    private static String httpPost(int port, String path, String body) throws Exception {
        return MockEngineTestSupport.httpPost(port, path, body);
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }

}
