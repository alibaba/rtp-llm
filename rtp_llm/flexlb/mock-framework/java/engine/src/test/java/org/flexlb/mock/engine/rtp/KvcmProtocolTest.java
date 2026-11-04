package org.flexlb.mock.engine.rtp;

import com.fasterxml.jackson.databind.*;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.sun.net.httpserver.HttpServer;
import io.grpc.*;
import java.net.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.BooleanSupplier;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class KvcmProtocolTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @TempDir Path directory;
    private RtpConfig config(String role, String endpoint, boolean events) throws Exception {
        var root = JSON.<ObjectNode>valueToTree(RtpNodeTest.configuration(role));
        var values = (ObjectNode)root.path("component").path("config");
        values.put("kv_cache_event_manager_endpoint",endpoint).put("reuse_cache",true)
            .put("kv_cache_event_host_ip_port","127.0.0.1:18080").put("kv_cache_event_instance_id","test-instance").put("kv_cache_event_instance_group","test-group");
        if (events) values.put("kv_cache_event_publisher_type","kvcm");
        else values.put("enable_remote_cache",true);
        var path = directory.resolve(role + ".json"); JSON.writeValue(path.toFile(),root); return RtpConfig.load(path);
    }
    private static void await(BooleanSupplier condition) throws Exception {
        long until = System.nanoTime() + TimeUnit.SECONDS.toNanos(8);
        while (!condition.getAsBoolean()) { assertTrue(System.nanoTime() < until,"Condition timed out"); Thread.sleep(10); }
    }
    static final class Meta implements AutoCloseable {
        final List<JsonNode> requests = new CopyOnWriteArrayList<>();
        final AtomicInteger registrations = new AtomicInteger();
        final AtomicBoolean expire = new AtomicBoolean();
        final AtomicBoolean expireItem = new AtomicBoolean();
        final HttpServer server;
        Meta() throws Exception {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1",0),0);
            server.createContext("/api/", exchange -> {
                try(exchange) {
                    var body = JSON.readTree(exchange.getRequestBody()); requests.add(body);
                    boolean registration = exchange.getRequestURI().getPath().equals("/api/registerInstance");
                    if (registration) registrations.incrementAndGet();
                    var response = JSON.createObjectNode();
                    response.putObject("header").putObject("status").put("code",!registration && expire.compareAndSet(true,false) ? 8 : 1);
                    if (!registration && expireItem.compareAndSet(true,false)) response.putArray("item_results").addObject().put("code",8);
                    response.put("extra_info","{\"rtp_remote_cache_enabled\":true}");
                    var bytes = JSON.writeValueAsBytes(response); exchange.sendResponseHeaders(200,bytes.length); exchange.getResponseBody().write(bytes);
                }
            }); server.start();
        }
        String endpoint() { return "http://127.0.0.1:" + server.getAddress().getPort(); }
        boolean has(String type) { return requests.stream().anyMatch(r -> r.path("events").findValuesAsText("event_type").contains(type)); }
        @Override public void close() { server.stop(0); }
    }
    @Test void nativeRegistrationSnapshotsMutationsRecoveryAndHostDownReflectRealCache() throws Exception {
        try(var meta = new Meta()) {
            try(var node = RtpNode.start(config("prefill",meta.endpoint(),true))) {
                await(() -> meta.has("EVENT_BLOCK_SNAPSHOT"));
                var registered = meta.requests.stream().filter(r -> r.has("model_deployment")).findFirst().orElseThrow();
                assertEquals("QT_PREFIX_MATCH",registered.path("default_query_type").asText());
                assertEquals(16,registered.path("block_size").intValue());
                assertEquals("FP16",registered.path("model_deployment").path("dtype").asText());
                try(var lease = node.cache().acquire(32,List.of()).lease()) { lease.publish(List.of(11L,22L),32); }
                await(() -> meta.has("EVENT_BLOCK_ADD"));
                var adds = meta.requests.stream().flatMap(r -> r.path("events").findValues("block_add").stream()).toList();
                assertTrue(adds.stream().anyMatch(a -> a.path("block_key").asText().equals("22")));
                try(var lease = node.cache().acquire(128,List.of()).lease()) { await(() -> meta.has("EVENT_BLOCK_DELETE")); }
                meta.expire.set(true); await(() -> meta.registrations.get() >= 2);
                meta.expireItem.set(true); await(() -> meta.registrations.get() >= 3);
            }
            assertTrue(meta.has("EVENT_HOST_DOWN"));
        }
    }
    @ParameterizedTest @ValueSource(ints = {600000, 0, -1})
    void controlOnlyKeepsHeartbeatsWithoutPublishingHbmKeys(int interval) throws Exception {
        try(var meta = new Meta()) {
            var root = JSON.<ObjectNode>valueToTree(RtpNodeTest.configuration("decode"));
            var values = (ObjectNode) root.path("component").path("config");
            values.put("enable_remote_cache", true).put("kv_cache_event_manager_endpoint", meta.endpoint())
                .put("kv_cache_event_instance_id", "control-only").put("kv_cache_event_snapshot_interval_ms", interval)
                .put("kv_cache_event_publisher_type", interval <= 0 ? "kvcm" : "none");
            var config = RtpConfig.load(root);
            values.put("enable_remote_cache", false).put("kv_cache_event_snapshot_interval_ms", 0);
            assertThrows(IllegalArgumentException.class, () -> RtpConfig.load(root));
            var remote = new org.flexlb.mock.engine.cache.remote.RemoteCache() {
                public Match match(List<Long> keys, int local, long deadline) { return () -> local; }
                public void read(Match match, List<Long> keys, long deadline) { throw new AssertionError("No reads expected"); }
                public int write(List<Long> keys, long deadline) { throw new AssertionError("No writes expected"); }
                public void close() {}
            };
            try (var node = RtpNode.start(config, 0, 0, RtpScheduler.factory(config), remote)) {
                await(() -> meta.has("EVENT_HEARTBEAT"));
                assertFalse(meta.has("EVENT_BLOCK_SNAPSHOT"));
                var registration = meta.requests.stream().filter(r -> r.path("events").findValuesAsText("event_type").contains("EVENT_NODE_REGISTER")).findFirst().orElseThrow();
                assertTrue(registration.path("events").get(0).path("node_register").path("mediums").isEmpty());
                assertEquals(false,((Map<?,?>)node.endpoints().get("kvcm")).get("payload_io"));
                assertEquals(true,((Map<?,?>)node.endpoints().get("kvcm")).get("remote_enabled"));
            }
        }
    }
    @Test void sharedReporterTracksHbmRemovalAndRestorationWithoutAnRtpScheduler() throws Exception {
        try (var meta = new Meta(); var cache = new org.flexlb.mock.engine.cache.KvCache(1, 4, 0,
                org.flexlb.mock.engine.cache.KvCache.EvictionPolicy.LRU, true, 2,
                org.flexlb.mock.engine.cache.KvCache.EvictionPolicy.LRU, new org.flexlb.mock.engine.cache.KvCache.Watermark(.25, .75), null);
             var reporter = new org.flexlb.mock.engine.cache.KvcmReporter(
                new org.flexlb.mock.engine.cache.KvcmReporter.Configuration(URI.create(meta.endpoint()), true, 600000),
                new org.flexlb.mock.engine.cache.KvcmReporter.Identity("shared", "test-group", "127.0.0.1:12345",
                    "full", "FP16", false, 1, 1, 1, "rtp_llm_hbm_1_full"), cache)) {
            await(() -> meta.has("EVENT_BLOCK_SNAPSHOT"));
            try (var lease = cache.acquire(1, List.of()).lease()) { lease.publish(List.of(99L), 1); }
            await(() -> meta.has("EVENT_BLOCK_ADD"));
            try (var pressure = cache.acquire(3, List.of()).lease()) {
                await(() -> meta.has("EVENT_BLOCK_DELETE"));
                assertEquals(Set.of(99L), cache.metadata().keys(org.flexlb.mock.engine.cache.CacheMetadata.Medium.HOST));
            }
            try (var restored = cache.acquire(1, List.of(99L)).lease()) { assertEquals(1, restored.reusedTokens()); }
            await(() -> meta.requests.stream().mapToLong(r -> r.path("events").findValues("block_add").size()).sum() >= 2);
        }
    }

    @Test void hostOnlyCacheRemainsVisibleToMasterButIsNeverReportedAsHbm() throws Exception {
        try (var meta = new Meta()) {
            var root = JSON.<ObjectNode>valueToTree(RtpNodeTest.configuration("prefill"));
            var values = (ObjectNode) root.path("component").path("config");
            values.put("kv_cache_event_publisher_type", "kvcm").put("kv_cache_event_manager_endpoint", meta.endpoint())
                .put("kv_cache_event_host_ip_port", "127.0.0.1:18080").put("kv_cache_event_instance_id", "host-only").put("reuse_cache", true)
                .put("enable_device_cache", false).put("enable_memory_cache", true).put("mock.cache.memory.blocks", 8);
            try (var node = RtpNode.start(RtpConfig.load(root))) {
                await(() -> meta.has("EVENT_BLOCK_SNAPSHOT"));
                try (var lease = node.cache().acquire(32, List.of()).lease()) { lease.publish(List.of(11L, 22L), 32); }
                var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
                try {
                    var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2, TimeUnit.SECONDS);
                    var first = rpc.getCacheStatus(CacheVersionPB.newBuilder().setLatestCacheVersion(-1).setNeedCacheKeys(true).build());
                    assertEquals(Set.of(11L, 22L), first.getCacheKeysMap().keySet());
                    assertEquals(16, first.getBlockSize());
                    var unchanged = rpc.getCacheStatus(CacheVersionPB.newBuilder().setLatestCacheVersion(first.getVersion()).setNeedCacheKeys(true).build());
                    assertEquals(first.getVersion(), unchanged.getVersion()); assertTrue(unchanged.getCacheKeysMap().isEmpty());
                    // A fresh snapshot must exclude Host-only keys as well as incremental reports.
                    meta.expire.set(true); await(() -> meta.registrations.get() >= 2);
                    await(() -> meta.requests.stream().filter(r -> r.path("events").findValuesAsText("event_type").contains("EVENT_BLOCK_SNAPSHOT")).count() >= 2);
                    assertFalse(meta.has("EVENT_BLOCK_ADD"));
                    for (var report : meta.requests) for (var snapshot : report.path("events").findValues("block_snapshot"))
                        assertTrue(snapshot.path("blocks").isEmpty());
                } finally { channel.shutdownNow().awaitTermination(2, TimeUnit.SECONDS); }
            }
        }
    }

    @Test void unimplementedPayloadIoAndRankShutdownNeverFabricateSuccess() throws Exception {
        var path = directory.resolve("native.json"); JSON.writeValue(path.toFile(),RtpNodeTest.configuration("prefill"));
        try(var node = RtpNode.start(RtpConfig.load(path))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2,TimeUnit.SECONDS);
                for(var op : List.of(RemoteOpType.REMOTE_OPERATION_READ, RemoteOpType.REMOTE_OPERATION_WRITE)) {
                    var result = rpc.executeFunction(FunctionRequestPB.newBuilder().setRemoteRequest(RemoteOperationRequestPB.newBuilder().setOp(op)).build()).getRemoteResponse();
                    assertEquals(RemoteTransferStatusPB.REMOTE_TRANSFER_STATUS_FAILED_BUFFER_SAFE,result.getTransferStatus()); assertTrue(result.getPayloadBufferSafe());
                }
                var failure = assertThrows(StatusRuntimeException.class, () -> rpc.executeFunction(
                    FunctionRequestPB.newBuilder().setShutdownRequest(ShutdownRequestPB.newBuilder().setBeginShutdown(true)).build()));
                assertEquals(Status.Code.UNIMPLEMENTED,failure.getStatus().getCode());

            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }
}
