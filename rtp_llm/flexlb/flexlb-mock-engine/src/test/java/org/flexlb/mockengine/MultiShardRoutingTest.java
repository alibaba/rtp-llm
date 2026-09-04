package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Multi-shard routing tests for the Java mock engine.
 *
 * <p>Verifies that multi-shard configurations (multiple prefill engine groups
 * serving different shards) route requests correctly, complete all requests
 * without leaks, and distribute load evenly across engines within each shard.
 *
 * <p>Since the mock engine cluster has no native shard concept, shards are
 * simulated by grouping engines on different port ranges and routing requests
 * to decode engines within the same shard via {@code RoleAddrPB} port hints.
 */
class MultiShardRoutingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP_CLIENT = HttpClient.newHttpClient();
    private static final int BASE_PORT = 63000;

    @TempDir
    Path tempDir;

    private MockEngineTestCluster cluster;
    private Map<Integer, JavaMockEngineCluster.FastRpcService> services;
    private List<List<JavaMockEngineCluster.FastRpcService>> prefillByShard;
    private List<List<JavaMockEngineCluster.FastRpcService>> decodeByShard;

    @AfterEach
    void tearDown() {
        if (cluster != null) {
            cluster.close();
        }
    }

    // ──────────── Test 1: Balanced 2x2 shard configuration ────────────

    @Test
    void balancedTwoShardRoutingCompletesAllRequestsWithNoLeak() throws Exception {
        MockPerformanceModel model = model("10");
        // 2 shards, each with 2 prefill + 2 decode engines (4P + 4D total)
        startMultiShardCluster(model, List.of(
                new ShardConfig(2, 2),
                new ShardConfig(2, 2)), BASE_PORT);

        int requestsPerShard = 20;
        int totalRequests = requestsPerShard * prefillByShard.size(); // 40

        // Enqueue requests round-robin across prefill engines within each shard
        long batchId = 10_000;
        int requestId = 1;
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            for (int i = 0; i < requestsPerShard; i++) {
                JavaMockEngineCluster.FastRpcService prefill = prefillEngines.get(i % prefillEngines.size());
                int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
                EngineRpcService.GenerateInputPB input = inputWithDecode(requestId++, 10, decodePort);
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(batchId++, slot(0, input)));
                assertEquals(0, response.getErrorsCount(),
                        "no errors expected for request " + (requestId - 1));
                assertEquals(1, response.getSuccessesCount(),
                        "request " + (requestId - 1) + " should be accepted");
            }
        }

        // Wait for all requests to complete
        awaitTotalCompleted(totalRequests, 10_000);

        // Assert no inflight leak on any engine
        assertAllInflightZero();

        // Assert no leak_detected flag on any engine
        for (JavaMockEngineCluster.FastRpcService service : services.values()) {
            assertFalse(service.isLeakDetected(),
                    "no leak should be detected on port " + service.getGrpcPort());
        }

        // Verify prefill load distribution: no single engine handles >50% (tolerance 55%)
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            long shardPrefillTotal = prefillEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getAcceptedCount)
                    .sum();
            assertEquals(requestsPerShard, shardPrefillTotal,
                    "shard " + shard + " prefill engines should accept " + requestsPerShard + " total");
            for (JavaMockEngineCluster.FastRpcService engine : prefillEngines) {
                double ratio = (double) engine.getAcceptedCount() / shardPrefillTotal;
                assertTrue(ratio <= 0.55,
                        "prefill engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " load (>55%)");
            }
        }

        // Verify decode engines received all requests and distributed load
        for (int shard = 0; shard < decodeByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            long shardDecodeTotal = decodeEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            assertEquals(requestsPerShard, shardDecodeTotal,
                    "shard " + shard + " decode engines should complete " + requestsPerShard + " total");
            for (JavaMockEngineCluster.FastRpcService engine : decodeEngines) {
                double ratio = (double) engine.getCompletedCount() / shardDecodeTotal;
                assertTrue(ratio <= 0.55,
                        "decode engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " decode load (>55%)");
            }
        }

        // Verify HTTP snapshot shows no leaks
        JsonNode snapshot = snapshot();
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port " + engine.get("port").asInt());
            assertFalse(engine.get("leak_detected").asBoolean(),
                    "snapshot leak_detected should be false for port " + engine.get("port").asInt());
        }
    }

    // ──────────── Test 2: Asymmetric P/D ratios per shard ────────────

    @Test
    void asymmetricPdRatioPerShardCompletesAllRequestsWithNoLeak() throws Exception {
        MockPerformanceModel model = model("10");
        // Shard 0: 2P + 4D, Shard 1: 2P + 2D
        startMultiShardCluster(model, List.of(
                new ShardConfig(2, 4),
                new ShardConfig(2, 2)), BASE_PORT);

        int requestsPerShard = 20;
        int totalRequests = requestsPerShard * prefillByShard.size(); // 40

        long batchId = 20_000;
        int requestId = 1;
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            for (int i = 0; i < requestsPerShard; i++) {
                JavaMockEngineCluster.FastRpcService prefill = prefillEngines.get(i % prefillEngines.size());
                int decodePort = decodeEngines.get(i % decodeEngines.size()).getGrpcPort();
                EngineRpcService.GenerateInputPB input = inputWithDecode(requestId++, 10, decodePort);
                EngineRpcService.EnqueueBatchResponsePB response =
                        enqueue(prefill, batch(batchId++, slot(0, input)));
                assertEquals(0, response.getErrorsCount(),
                        "no errors expected for shard " + shard + " request " + i);
                assertEquals(1, response.getSuccessesCount(),
                        "request " + requestId + " should be accepted");
            }
        }

        awaitTotalCompleted(totalRequests, 10_000);
        assertAllInflightZero();

        // Verify prefill load distribution within each shard
        for (int shard = 0; shard < prefillByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> prefillEngines = prefillByShard.get(shard);
            long shardPrefillTotal = prefillEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getAcceptedCount)
                    .sum();
            assertEquals(requestsPerShard, shardPrefillTotal,
                    "shard " + shard + " prefill should accept " + requestsPerShard);
            for (JavaMockEngineCluster.FastRpcService engine : prefillEngines) {
                double ratio = (double) engine.getAcceptedCount() / shardPrefillTotal;
                assertTrue(ratio <= 0.6,
                        "prefill engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " load (>60%)");
            }
        }

        // Verify decode load distribution within each shard
        for (int shard = 0; shard < decodeByShard.size(); shard++) {
            List<JavaMockEngineCluster.FastRpcService> decodeEngines = decodeByShard.get(shard);
            long shardDecodeTotal = decodeEngines.stream()
                    .mapToLong(JavaMockEngineCluster.FastRpcService::getCompletedCount)
                    .sum();
            assertEquals(requestsPerShard, shardDecodeTotal,
                    "shard " + shard + " decode should complete " + requestsPerShard);
            for (JavaMockEngineCluster.FastRpcService engine : decodeEngines) {
                double ratio = (double) engine.getCompletedCount() / shardDecodeTotal;
                assertTrue(ratio <= 0.55,
                        "decode engine port " + engine.getGrpcPort()
                                + " handles " + String.format("%.1f%%", ratio * 100)
                                + " of shard " + shard + " decode load (>55%)");
            }
        }

        // Verify HTTP snapshot
        JsonNode snapshot = snapshot();
        assertEquals(services.size(), snapshot.size());
        for (JsonNode engine : snapshot) {
            assertEquals(0, engine.get("inflight").asInt(),
                    "snapshot inflight should be 0 for port " + engine.get("port").asInt());
            assertFalse(engine.get("leak_detected").asBoolean(),
                    "snapshot leak_detected should be false for port " + engine.get("port").asInt());
        }
    }

    // ──────────── Cluster setup ────────────

    private record ShardConfig(int nPrefill, int nDecode) {}

    /**
     * Start a multi-shard cluster. All prefill engines are created first
     * (contiguous ports), then all decode engines, so that shard prefill
     * ports don't collide with another shard's decode ports.
     */
    private void startMultiShardCluster(MockPerformanceModel model,
                                         List<ShardConfig> shards, int basePort) throws IOException {
        int prefillCount = shards.stream().mapToInt(ShardConfig::nPrefill).sum();
        int decodeCount = shards.stream().mapToInt(ShardConfig::nDecode).sum();
        cluster = MockEngineTestCluster.start(
                model, basePort, prefillCount, decodeCount);
        services = cluster.services();
        prefillByShard = new ArrayList<>();
        decodeByShard = new ArrayList<>();

        int prefillOffset = 0;
        int decodeOffset = 0;
        for (ShardConfig shard : shards) {
            prefillByShard.add(List.copyOf(cluster.prefills().subList(
                    prefillOffset, prefillOffset + shard.nPrefill)));
            decodeByShard.add(List.copyOf(cluster.decodes().subList(
                    decodeOffset, decodeOffset + shard.nDecode)));
            prefillOffset += shard.nPrefill;
            decodeOffset += shard.nDecode;
        }
    }

    // ──────────── Polling helpers ────────────

    private long totalCompleted() {
        return cluster.totalCompleted();
    }

    private void awaitTotalCompleted(int expected, long timeoutMs) throws InterruptedException {
        cluster.awaitCompleted(expected, timeoutMs);
    }

    private void assertAllInflightZero() {
        cluster.assertAllInflightZero();
    }

    // ──────────── HTTP helpers ────────────

    private JsonNode snapshot() throws Exception {
        return cluster.snapshot();
    }

    // ──────────── Model helper ────────────

    private MockPerformanceModel model(String formula) throws Exception {
        return MockEngineTestSupport.performanceModel(tempDir, formula);
    }

}
