package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;

import java.net.InetSocketAddress;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.*;

/** Real Mock JVM + gRPC requests + HTTP protocol peer; no real KVCM server. */
@Timeout(45)
class KvcmEngineProcessTest {
    @TempDir Path directory;
    private static final ObjectMapper JSON = new ObjectMapper();
    private static void until(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) Thread.sleep(10);
        assertTrue(condition.getAsBoolean());
    }

    @Test void runtimeEnvironmentPublishesRequestKeysEvictionsAndShutdownPerEngine() throws Exception {
        int base = Integer.parseInt(System.getenv().getOrDefault("FLEXLB_PORT_BASE", "62400")) + 10;
        String prefill = "127.0.0.2:" + (base - 1), decode = "127.0.0.2:" + base;
        Map<String, Set<Long>> owners = new ConcurrentHashMap<>();
        Set<String> down = ConcurrentHashMap.newKeySet();
        HttpServer peer = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        peer.createContext("/api/", exchange -> {
            JsonNode root = JSON.readTree(exchange.getRequestBody());
            String host = root.path("host_ip_port").asText();
            if (!host.isEmpty()) {
                Set<Long> keys = owners.computeIfAbsent(host, ignored -> ConcurrentHashMap.newKeySet());
                for (JsonNode event : root.path("events")) {
                    switch (event.path("event_type").asText()) {
                        case "EVENT_BLOCK_SNAPSHOT" -> {
                            keys.clear();
                            event.path("block_snapshot").path("blocks").forEach(block ->
                                    keys.add(Long.parseLong(block.path("block_key").asText())));
                        }
                        case "EVENT_BLOCK_ADD" -> keys.add(Long.parseLong(event.path("block_add").path("block_key").asText()));
                        case "EVENT_BLOCK_DELETE" -> keys.remove(Long.parseLong(event.path("block_delete").path("block_key").asText()));
                        case "EVENT_HOST_DOWN" -> { keys.clear(); down.add(host); }
                        default -> { }
                    }
                }
            }
            byte[] response = "{\"header\":{\"status\":{\"code\":\"OK\"}}}".getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            try (var out = exchange.getResponseBody()) { out.write(response); }
            exchange.close();
        });
        peer.start();
        MockEngineTestSupport.performanceModel(directory, "5", 0.1);
        Path performance, master;
        try (var files = Files.list(directory)) {
            performance = files.filter(path -> path.getFileName().toString().startsWith("performance-")).findFirst().orElseThrow();
        }
        try (var files = Files.list(directory)) {
            master = files.filter(path -> path.getFileName().toString().startsWith("master-")).findFirst().orElseThrow();
        }
        Path log = directory.resolve("mock.log");
        Process process = null;
        ManagedChannel channel = null;
        try (HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(1)).build()) {
            ProcessBuilder builder = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin/java").toString(),
                    "-cp", System.getProperty("java.class.path"), JavaMockEngineCluster.class.getName(),
                    "--whale", "true", "--whale-bundle", "true", "--unique-engine-ips", "false",
                    "--n-prefill", "1", "--n-decode", "1", "--host", "127.0.0.2", "--bind-host", "127.0.0.2",
                    "--base-grpc-port", Integer.toString(base), "--event-loop-threads", "2", "--completion-threads", "2",
                    "--performance", performance.toString(),
                    "--master-config", master.toString(),
                    "--endpoint-file", directory.resolve("endpoints.json").toString())
                    .redirectErrorStream(true).redirectOutput(log.toFile());
            builder.environment().putAll(Map.of("KV_CACHE_EVENT_PUBLISHER_TYPE", "kvcm",
                    "KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://127.0.0.1:" + peer.getAddress().getPort(),
                    "KV_CACHE_EVENT_INSTANCE_ID", "process-test", "KV_CACHE_EVENT_INSTANCE_GROUP", "process-test",
                    "MOCK_KVCM_FLUSH_MS", "5"));
            builder.environment().remove("KV_CACHE_EVENT_HOST_IP_PORT");
            process = builder.start();
            until(() -> owners.containsKey(prefill) && owners.containsKey(decode));
            channel = ManagedChannelBuilder.forAddress("127.0.0.2", base).usePlaintext().build();
            var stub = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(10, TimeUnit.SECONDS);
            var status = stub.getWorkerStatus(EngineRpcService.StatusVersionPB.getDefaultInstance());
            assertEquals(1024, status.getBlockSize());
            assertEquals(EngineRpcService.KvCacheGroupModePB.KV_CACHE_GROUP_MODE_FULL_ATTENTION_ONLY,
                    status.getKvCacheGroupMode());
            assertEquals(0, status.getBlockHashLookaheadTokens());
            assertEquals(0, status.getCacheMatchRollbackBlocks());
            var inputBuilder = MockEngineTestSupport.inputWithDecode(9001, 2049, base + 1, 2).toBuilder();
            inputBuilder.getGenerateConfigBuilder().setReuseCache(true);
            var input = inputBuilder.build();
            var ack = stub.enqueueBatch(MockEngineTestSupport.batch(9001, MockEngineTestSupport.slot(0, input)));
            assertEquals(0, ack.getErrorsCount());
            boolean completed = false;
            var responses = stub.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder().setRequestId(9001).build());
            while (responses.hasNext()) {
                var frame = responses.next();
                assertEquals(0, frame.getErrorInfo().getErrorCode().getNumber());
                completed |= frame.getFlattenOutput().getFinishedList().contains(true);
            }
            assertTrue(completed);
            until(() -> owners.get(prefill).size() == 2 && owners.get(decode).size() == 2);
            var snapshot = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.2:" + (base - 1) + "/snapshot"))
                    .timeout(Duration.ofSeconds(2)).GET().build(), HttpResponse.BodyHandlers.ofString());
            assertEquals(200, snapshot.statusCode());
            assertTrue(snapshot.body().contains("\"kvcm\""));
            Long evicted = owners.get(prefill).iterator().next();
            var eviction = http.send(HttpRequest.newBuilder(URI.create("http://127.0.0.2:" + (base - 1) + "/cache_evict"))
                    .timeout(Duration.ofSeconds(2)).header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(Map.of("engine", "prefill-0", "keys", Set.of(evicted)))))
                    .build(), HttpResponse.BodyHandlers.ofString());
            assertEquals(200, eviction.statusCode());
            until(() -> owners.get(prefill).size() == 1);
            assertEquals(2, owners.get(decode).size());
            process.destroy();
            assertTrue(process.waitFor(8, TimeUnit.SECONDS));
            assertEquals(Set.of(prefill, decode), down);
        } catch (Throwable error) {
            if (Files.exists(log)) System.err.println(Files.readString(log));
            throw error;
        } finally {
            if (channel != null) channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            if (process != null && process.isAlive()) {
                process.destroyForcibly(); process.waitFor(3, TimeUnit.SECONDS);
            }
            peer.stop(0);
        }
    }
}
