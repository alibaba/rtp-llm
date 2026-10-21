package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.protobuf.CodedOutputStream;
import com.sun.net.httpserver.HttpServer;
import io.grpc.ServerBuilder;
import io.grpc.ServerServiceDefinition;
import io.grpc.stub.ServerCalls;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

/** Behavior ports from 969cb846e publisher/tree tests; HTTP peers are not a real KVCM server. */
class KvcmRealParityTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String OK = "{\"header\":{\"status\":{\"code\":\"OK\"}}}";
    private static void until(BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) Thread.sleep(5);
        assertTrue(condition.getAsBoolean());
    }
    static KvcmEventPublisher.Config config(URI uri, List<KvcmEventPublisher.Group> groups) {
        return new KvcmEventPublisher.Config(uri, "g", "model", "mock", 64, 32, 5, 20,
                10000, 500, 5, 50, "ST_EVENT_REPORT_L1P5", groups, null, 2000, 300, true);
    }
    private static final List<KvcmEventPublisher.Group> GROUPS = List.of(
            new KvcmEventPublisher.Group("full0", "FULL", true, 4096),
            new KvcmEventPublisher.Group("full1", "FULL", true, 8192),
            new KvcmEventPublisher.Group("swa", "SWA", true, 2048),
            new KvcmEventPublisher.Group("linear", "LINEAR", true, 1024));
    private static boolean event(JsonNode node, String type) {
        return node.path("events").path(0).path("event_type").asText().equals(type);
    }
    private static final class Peer implements AutoCloseable {
        final HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        final java.util.concurrent.ExecutorService executor = Executors.newCachedThreadPool();
        final List<JsonNode> requests = new CopyOnWriteArrayList<>();
        volatile Function<JsonNode, String> reply = ignored -> OK;
        Peer() throws Exception {
            server.setExecutor(executor);
            server.createContext("/api/", exchange -> {
                JsonNode request = JSON.readTree(exchange.getRequestBody());
                requests.add(request);
                byte[] body = reply.apply(request).getBytes(java.nio.charset.StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(200, body.length);
                try (var out = exchange.getResponseBody()) { out.write(body); }
            });
            server.start();
        }
        URI uri() { return URI.create("http://127.0.0.1:" + server.getAddress().getPort()); }
        long count(String type) { return requests.stream().filter(n -> event(n, type)).count(); }
        public void close() { server.stop(0); executor.shutdownNow(); }
    }
    @Test void filtersLinearAndPublishesPerGroupResidencyInsteadOfFlattening() throws Exception {
        var cache = new MockLruBlockCache(8);
        cache.admit(List.of(1L, 2L));
        var requests = new CopyOnWriteArrayList<JsonNode>();
        KvcmEventPublisher.Transport peer = (route, body) -> { requests.add(JSON.readTree(body)); return true; };
        try (var publisher = new KvcmEventPublisher(config(URI.create("http://localhost:1"), GROUPS), cache, "host:1", 512, () -> true, peer)) {
            publisher.start();
            until(() -> publisher.status().get("state").equals("READY"));
            JsonNode registration = requests.getFirst();
            assertEquals("QT_PREFIX_MATCH", registration.path("default_query_type").asText());
            assertEquals(3, registration.path("location_spec_infos").size());
            assertEquals(3, requests.stream().filter(n -> event(n, "EVENT_BLOCK_SNAPSHOT")).findFirst().orElseThrow()
                    .path("events").path(0).path("block_snapshot").path("blocks").path(0).path("specs").size());
            assertTrue(publisher.setGroupResident(1, "full1", false));
            until(() -> requests.stream().flatMap(n -> java.util.stream.StreamSupport.stream(n.path("events").spliterator(), false)).anyMatch(e ->
                    e.path("block_delete").path("spec_names").toString().contains("full1")));
            assertEquals(0, cache.prefixHitBlocks(List.of(1L, 2L)), "partial group is not reusable");
            assertEquals(List.of(0, 2), cache.componentSnapshot().groups().get(1L));
            assertFalse(publisher.setGroupResident(1, "full1", false));
            assertThrows(IllegalArgumentException.class, () -> publisher.setGroupResident(1, "linear", true));
            assertTrue(publisher.setGroupResident(1, "full1", true));
            until(() -> requests.stream().flatMap(n -> java.util.stream.StreamSupport.stream(n.path("events").spliterator(), false)).anyMatch(e ->
                    e.path("block_add").path("specs").toString().contains("full1")));
            assertEquals(2, cache.prefixHitBlocks(List.of(1L, 2L)));
            cache.evict(List.of(1L));
            assertFalse(cache.componentSnapshot().groups().containsKey(1L));
        }
    }
    @Test void feedbackParsingPreservesHintsEvenWhenHeaderIsOk() {
        var f = KvcmEventPublisher.parseFeedback("{\"header\":{\"status\":{\"code\":1}},\"snapshot_required\":true,\"retry_after_ms\":\"500\"}");
        assertTrue(f.valid() && f.ok() && f.snapshotRequired()); assertEquals(500, f.retryAfterMs());
        var nested = KvcmEventPublisher.parseFeedback("{\"header\":{\"status\":{\"code\":\"OK\"}},\"item_results\":[{\"status\":{\"code\":\"NODE_NOT_REGISTERED\"}}]}");
        assertFalse(nested.ok()); assertTrue(nested.registrationRequired());
        assertFalse(KvcmEventPublisher.parseFeedback("{\"header\":{\"status\":{\"code\":1}},\"snapshot_required\":0}").valid());
        assertFalse(KvcmEventPublisher.parseFeedback("{\"header\":{\"status\":{\"code\":1}},\"retry_after_ms\":-1}").valid());
    }
    @Test void slowSnapshotKeepsHeartbeatsAndHostDownIsLast() throws Exception {
        try (var peer = new Peer()) {
            var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
            peer.reply = request -> {
                if (event(request, "EVENT_BLOCK_SNAPSHOT")) {
                    entered.countDown();
                    try { assertTrue(release.await(3, TimeUnit.SECONDS)); } catch (InterruptedException error) { throw new RuntimeException(error); }
                }
                return OK;
            };
            try (var publisher = new KvcmEventPublisher(config(peer.uri(), GROUPS), new MockLruBlockCache(8), "host:1", 512, () -> true, null)) {
                publisher.start(); assertTrue(entered.await(2, TimeUnit.SECONDS));
                try { until(() -> peer.count("EVENT_HEARTBEAT") >= 3); }
                finally { release.countDown(); }
                until(() -> publisher.status().get("state").equals("READY"));
            }
            assertTrue(event(peer.requests.getLast(), "EVENT_HOST_DOWN"));
            long count = peer.requests.size(); Thread.sleep(60); assertEquals(count, peer.requests.size());
        }
    }
    @Test void successfulHeartbeatHintTriggersSnapshotWithoutWaitingForPeriodicTimer() throws Exception {
        try (var peer = new Peer()) {
            var hint = new AtomicBoolean();
            peer.reply = request -> event(request, "EVENT_HEARTBEAT") && hint.getAndSet(false)
                    ? "{\"header\":{\"status\":{\"code\":1}},\"snapshot_required\":true}" : OK;
            try (var publisher = new KvcmEventPublisher(config(peer.uri(), GROUPS), new MockLruBlockCache(8), "host:1", 512, () -> true, null)) {
                publisher.start(); until(() -> publisher.status().get("state").equals("READY"));
                hint.set(true); until(() -> peer.count("EVENT_BLOCK_SNAPSHOT") >= 2);
                assertEquals(0L, publisher.status().get("failures"));
            }
        }
    }
    @Test void retryAfterThrottlesSnapshotsButNotHeartbeats() throws Exception {
        try (var peer = new Peer()) {
            var once = new AtomicBoolean(true);
            var times = new CopyOnWriteArrayList<Long>();
            peer.reply = request -> {
                if (event(request, "EVENT_BLOCK_SNAPSHOT")) {
                    times.add(System.nanoTime());
                    if (once.getAndSet(false)) return "{\"header\":{\"status\":{\"code\":14}},\"retry_after_ms\":200}";
                }
                return OK;
            };
            try (var publisher = new KvcmEventPublisher(config(peer.uri(), GROUPS), new MockLruBlockCache(8), "host:1", 512, () -> true, null)) {
                publisher.start(); until(() -> times.size() >= 2 && publisher.status().get("state").equals("READY"));
                assertTrue(TimeUnit.NANOSECONDS.toMillis(times.get(1) - times.get(0)) >= 180);
                assertTrue(peer.count("EVENT_HEARTBEAT") >= 3);
                var snapshots = peer.requests.stream().filter(n -> event(n, "EVENT_BLOCK_SNAPSHOT")).toList();
                assertEquals(snapshots.get(0), snapshots.get(1), "valid server backoff retries identical pending snapshot");
            }
        }
    }
    @Test void registrationFailureAlsoRespectsServerBackoff() throws Exception {
        try (var peer = new Peer()) {
            var times = new CopyOnWriteArrayList<Long>();
            peer.reply = request -> {
                if (request.has("instance_group")) {
                    times.add(System.nanoTime());
                    if (times.size() == 1) return "{\"header\":{\"status\":{\"code\":8}},\"retry_after_ms\":150}";
                }
                return OK;
            };
            try (var publisher = new KvcmEventPublisher(config(peer.uri(), GROUPS), new MockLruBlockCache(8), "host:1", 512, () -> true, null)) {
                publisher.start(); until(() -> times.size() >= 2 && publisher.status().get("state").equals("READY"));
                assertTrue(TimeUnit.NANOSECONDS.toMillis(times.get(1) - times.get(0)) >= 130);
            }
        }
    }
    @Test void discoveryInvalidatesLeaderAndFixedEndpointNeverUsesDiscovery() throws Exception {
        var base = config(URI.create("http://localhost:1"), GROUPS);
        var cfg = new KvcmEventPublisher.Config(null, base.group(), base.instance(), base.model(), 64, 32, 5, 20, 10000,
                500, 5, 50, base.storageType(), GROUPS, new KvcmEventPublisher.Discovery(false, "127.0.0.1:6381"), 2000, 300, true);
        var calls = new AtomicInteger();
        try (var resolver = new KvcmEndpointResolver(cfg, "model_512", () -> List.of("bad:1", "good:6381"), (target, instance, timeout) -> {
            if (target.startsWith("bad")) throw new IllegalStateException("unreachable");
            assertEquals("model_512", instance); return URI.create("http://127.0.0.1:" + (7000 + calls.incrementAndGet()));
        })) {
            URI first = resolver.endpoint(); assertEquals(first, resolver.endpoint()); assertEquals(1, calls.get());
            resolver.invalidate(first); assertNotEquals(first, resolver.endpoint());
        }
        try (var fixed = new KvcmEndpointResolver(config(URI.create("http://localhost:6382"), GROUPS), "id",
                () -> { throw new AssertionError("fixed endpoint must not discover"); }, (a,b,c) -> null)) {
            assertEquals(6382, fixed.endpoint().getPort()); fixed.invalidate(fixed.endpoint()); assertEquals(6382, fixed.endpoint().getPort());
        }
    }
    private static byte[] message(java.util.function.Consumer<CodedOutputStream> writer) throws Exception {
        var bytes = new ByteArrayOutputStream(); var out = CodedOutputStream.newInstance(bytes); writer.accept(out); out.flush(); return bytes.toByteArray();
    }
    @Test void discoveryUsesActualGrpcMethodAndPublishedProtobufFieldNumbers() throws Exception {
        byte[] header = new byte[]{10, 2, 8, 1};
        byte[] leader = message(out -> { try { out.writeString(2, "127.0.0.1"); out.writeInt32(4, 6382); } catch (Exception e) { throw new RuntimeException(e); } });
        byte[] response = message(out -> { try { out.writeByteArray(1, header); out.writeByteArray(4, leader); } catch (Exception e) { throw new RuntimeException(e); } });
        var service = ServerServiceDefinition.builder("kv_cache_manager.proto.meta.MetaService")
                .addMethod(KvcmEndpointResolver.METHOD, ServerCalls.asyncUnaryCall((byte[] request, io.grpc.stub.StreamObserver<byte[]> observer) -> {
                    observer.onNext(response); observer.onCompleted();
                })).build();
        var server = ServerBuilder.forPort(0).addService(service).build().start();
        try { assertEquals(URI.create("http://127.0.0.1:6382"), KvcmEndpointResolver.queryLeader("127.0.0.1:" + server.getPort(), "model", 2000)); }
        finally { server.shutdownNow().awaitTermination(2, TimeUnit.SECONDS); }
        assertThrows(IllegalArgumentException.class, () -> KvcmEndpointResolver.parseLeader(new byte[0]));
    }
    @Test void componentRecomputationRestoresOnlyMissingSpecsAndLocalReuse() {
        var cache = new MockLruBlockCache(8); cache.configurePublicationGroups(3);
        var events = new java.util.ArrayList<MockLruBlockCache.Mutation>(); cache.setMutationListener(events::add);
        cache.admit(List.of(1L)); assertEquals(3, events.size());
        cache.setGroupResident(1, 1, false); assertEquals(0, cache.prefixHitBlocks(List.of(1L)));
        assertTrue(cache.admit(List.of(1L))); assertEquals(1, cache.prefixHitBlocks(List.of(1L)));
        assertEquals(5, events.size()); assertEquals(1, events.getLast().groupIndex()); assertTrue(events.getLast().added());
        cache.admit(List.of(1L)); assertEquals(5, events.size(), "complete duplicate must not republish");
    }
    @Test void unsignedRetryAfterIsClampedRatherThanInvalidatingRegistration() {
        var response = KvcmEventPublisher.parseFeedback("{\"header\":{\"status\":{\"code\":1}},\"retry_after_ms\":\"18446744073709551615\"}");
        assertTrue(response.valid()); assertEquals(Long.MAX_VALUE, response.retryAfterMs());
    }
    @Test void packagedVipserverApiMatchesTheDiscoveryAdapter() throws Exception {
        Class<?> client;
        try { client = Class.forName("com.taobao.vipserver.client.core.VIPClient", false, getClass().getClassLoader()); }
        catch (ClassNotFoundException absent) { org.junit.jupiter.api.Assumptions.assumeTrue(false, "private discovery profile not enabled"); return; }
        assertTrue(java.lang.reflect.Modifier.isStatic(client.getMethod("srvHosts", String.class).getModifiers()));
        Class<?> host = Class.forName("com.taobao.vipserver.client.core.Host", false, getClass().getClassLoader());
        assertEquals(String.class, host.getMethod("getIp").getReturnType());
        assertNotNull(host.getMethod("getPort"));
    }
    @Test void nativeIdentityAndModelDescriptorComeFromExplicitConfiguration() {
        var env = new java.util.HashMap<>(Map.of("KV_CACHE_EVENT_PUBLISHER_TYPE", "kvcm", "KV_CACHE_EVENT_INSTANCE_ID", "native_512",
                "KV_CACHE_EVENT_INSTANCE_GROUP", "g", "KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://localhost:6382",
                "MOCK_KVCM_DTYPE", "BF16", "MOCK_KVCM_TP_SIZE", "4", "MOCK_KVCM_DP_SIZE", "2", "MOCK_KVCM_DP_RANK", "1"));
        var config = KvcmEventPublisher.Config.fromEnv(env);
        assertFalse(config.appendBlockSize()); assertEquals(4, config.metadata().tp());
        assertEquals(1, config.metadata().dpRank()); assertEquals("BF16", config.metadata().dtype());
        env.put("MOCK_KVCM_APPEND_BLOCK_SIZE", "true"); assertTrue(KvcmEventPublisher.Config.fromEnv(env).appendBlockSize());
        env.put("MOCK_KVCM_PP_SIZE", "2"); assertThrows(IllegalArgumentException.class, () -> KvcmEventPublisher.Config.fromEnv(env));
    }
    @Test void environmentAllowsRealDiscoveryAndAlreadySuffixedInstance() {
        var cfg = KvcmEventPublisher.Config.fromEnv(Map.of("KV_CACHE_EVENT_PUBLISHER_TYPE", "kvcm", "KV_CACHE_EVENT_INSTANCE_ID", "model_512",
                "RECO_INSTANCE_GROUP", "group", "RECO_ENABLE_VIPSERVER", "1", "RECO_VIPSERVER_DOMAIN", "domain",
                "MOCK_KVCM_APPEND_BLOCK_SIZE", "false"));
        assertNull(cfg.endpoint()); assertTrue(cfg.discovery().vipserver()); assertFalse(cfg.appendBlockSize());
        assertEquals("ST_EVENT_REPORT_L1P5", cfg.storageType()); assertEquals(1000, cfg.heartbeatMs());
    }
}
