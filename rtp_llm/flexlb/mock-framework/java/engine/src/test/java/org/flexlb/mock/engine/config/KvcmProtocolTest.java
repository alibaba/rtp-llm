package org.flexlb.mock.engine.config;

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
import org.flexlb.mock.engine.execution.RtpNode;
import org.flexlb.mock.engine.scheduler.RtpNodeTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class KvcmProtocolTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    @TempDir Path directory;
    private RtpConfig config(String role, String endpoint) throws Exception {
        var root = JSON.<ObjectNode>valueToTree(RtpNodeTest.configuration(role));
        var values = (ObjectNode)root.path("component").path("config");
        values.put("kv_cache_event_manager_endpoint",endpoint).put("reuse_cache",true)
            .put("kv_cache_event_host_ip_port","127.0.0.1:18080").put("kv_cache_event_instance_id","test-instance").put("kv_cache_event_instance_group","test-group");
        values.put("kv_cache_event_publisher_type","kvcm");
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
            try(var node = RtpNode.start(config("prefill",meta.endpoint()))) {
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
    @Test void unimplementedPayloadIoAndRankShutdownNeverFabricateSuccess() throws Exception {
        var path = directory.resolve("native.json"); JSON.writeValue(path.toFile(),RtpNodeTest.configuration("prefill"));
        try(var node = RtpNode.start(RtpConfig.load(path))) {
            var channel = ManagedChannelBuilder.forTarget(node.rpcAddress()).usePlaintext().build();
            try {
                var rpc = RpcServiceGrpc.newBlockingStub(channel).withDeadlineAfter(2,TimeUnit.SECONDS);
                for(var op : List.of(RemoteOpType.REMOTE_OPERATION_READ, RemoteOpType.REMOTE_OPERATION_WRITE)) {
                    var result = rpc.executeFunction(FunctionRequestPB.newBuilder().setRemoteRequest(RemoteOperationRequestPB.newBuilder().setOp(op)).build()).getRemoteResponse();
                    assertEquals(RemoteTransferStatusPB.REMOTE_TRANSFER_STATUS_FAILED,result.getTransferStatus());
                }
                var failure = assertThrows(StatusRuntimeException.class, () -> rpc.executeFunction(
                    FunctionRequestPB.getDefaultInstance()));
                assertEquals(Status.Code.UNIMPLEMENTED,failure.getStatus().getCode());

            } finally { channel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }
}
