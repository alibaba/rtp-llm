package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.*;

class KvcmEventPublisherTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final String OK = "{\"header\":{\"status\":{\"code\":\"OK\"}}}";

    private static KvcmEventPublisher.Config config(URI endpoint, int capacity) {
        return new KvcmEventPublisher.Config(endpoint, "test-group", "test-deployment", "mock-model",
                capacity, 16, 5, 30, 10000, 500, 5, 50);
    }
    private static KvcmEventPublisher.Config config(int capacity) {
        return config(URI.create("http://127.0.0.1:1"), capacity);
    }
    private static void until(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) Thread.sleep(5);
        assertTrue(condition.getAsBoolean(), "publisher did not converge before deadline");
    }

    /** Protocol test peer, not a KVCM server or a substitute for real-server acceptance. */
    private static final class Peer implements KvcmEventPublisher.Transport {
        final List<JsonNode> requests = new CopyOnWriteArrayList<>();
        final Set<Long> keys = java.util.concurrent.ConcurrentHashMap.newKeySet();
        final AtomicBoolean rejectHeartbeat = new AtomicBoolean();
        final AtomicBoolean loseSnapshotAck = new AtomicBoolean();
        final AtomicBoolean loseMutationAck = new AtomicBoolean();
        volatile CountDownLatch snapshotEntered, snapshotRelease;
        @Override public boolean post(String route, String body) throws Exception {
            JsonNode root = JSON.readTree(body);
            requests.add(root);
            if (route.equals("/api/registerInstance")) return true;
            for (JsonNode event : root.path("events")) {
                String type = event.path("event_type").asText();
                switch (type) {
                    case "EVENT_BLOCK_SNAPSHOT" -> {
                        if (snapshotEntered != null) {
                            snapshotEntered.countDown();
                            assertTrue(snapshotRelease.await(3, TimeUnit.SECONDS));
                        }
                        keys.clear();
                        for (JsonNode block : event.path("block_snapshot").path("blocks"))
                            keys.add(Long.valueOf(block.path("block_key").asText()));
                        if (loseSnapshotAck.getAndSet(false)) return false;
                    }
                    case "EVENT_BLOCK_ADD" -> {
                        keys.add(Long.valueOf(event.path("block_add").path("block_key").asText()));
                        if (loseMutationAck.getAndSet(false)) return false;
                    }
                    case "EVENT_BLOCK_DELETE" -> keys.remove(Long.valueOf(event.path("block_delete").path("block_key").asText()));
                    case "EVENT_HEARTBEAT" -> { if (rejectHeartbeat.getAndSet(false)) return false; }
                    case "EVENT_HOST_DOWN" -> keys.clear();
                    default -> { }
                }
            }
            return true;
        }
        List<JsonNode> events(String type) {
            return requests.stream().flatMap(root -> {
                List<JsonNode> result = new ArrayList<>();
                root.path("events").forEach(event -> { if (event.path("event_type").asText().equals(type)) result.add(event); });
                return result.stream();
            }).toList();
        }
        List<JsonNode> snapshots() {
            return requests.stream().filter(root -> root.path("events").path(0).path("event_type").asText()
                    .equals("EVENT_BLOCK_SNAPSHOT")).toList();
        }
    }

    @Test void disabledAndInvalidConfigurationAndResponseContracts() {
        assertNull(KvcmEventPublisher.Config.fromEnv(Map.of()));
        assertThrows(IllegalArgumentException.class, () -> KvcmEventPublisher.Config.fromEnv(Map.of("KV_CACHE_EVENT_PUBLISHER_TYPE", "kvcm")));
        var env = new java.util.HashMap<>(Map.of("KV_CACHE_EVENT_PUBLISHER_TYPE", "kvcm",
                "KV_CACHE_EVENT_MANAGER_ENDPOINT", "localhost:56020/", "RECO_INSTANCE_GROUP", "group",
                "BIZ_NAME", "biz", "DEPLOYMENT_NAME", "dep"));
        var parsed = KvcmEventPublisher.Config.fromEnv(env);
        assertEquals(URI.create("http://localhost:56020"), parsed.endpoint());
        assertEquals("biz/dep", parsed.instance());
        env.put("MOCK_KVCM_STORAGE_TYPE", "ST_EVENT_REPORT_L1P5");
        assertEquals("ST_EVENT_REPORT_L1P5", KvcmEventPublisher.Config.fromEnv(env).storageType());
        env.put("MOCK_KVCM_STORAGE_TYPE", "unknown");
        assertThrows(IllegalArgumentException.class, () -> KvcmEventPublisher.Config.fromEnv(env));
        env.remove("MOCK_KVCM_STORAGE_TYPE");
        env.put("KV_CACHE_EVENT_HOST_IP_PORT", "shared:1");
        assertThrows(IllegalArgumentException.class, () -> KvcmEventPublisher.Config.fromEnv(env));
        assertTrue(KvcmEventPublisher.responseOk(OK));
        assertTrue(KvcmEventPublisher.responseOk("{\"header\":{\"status\":{\"code\":1}},\"itemResults\":[{\"status\":{\"code\":\"1\"}}]}"));
        assertFalse(KvcmEventPublisher.responseOk("{}"));
        assertFalse(KvcmEventPublisher.responseOk("{\"header\":{\"status\":{\"code\":\"OK\"}},\"item_results\":[{\"code\":0}]}"));
        assertFalse(KvcmEventPublisher.responseOk("invalid"));
    }

    @Test void logicalTransitionsExcludeHitsDuplicatesAndReferencedEviction() {
        var cache = new MockLruBlockCache(16);
        var mutations = new ArrayList<MockLruBlockCache.Mutation>();
        cache.setMutationListener(mutations::add);
        cache.admit(List.of(1L, 2L));
        cache.admit(List.of(1L, 2L));
        cache.prefixHitBlocks(List.of(1L, 2L));
        var lease = cache.pinExisting(List.of(1L));
        assertFalse(cache.evict(List.of(1L)));
        assertEquals(2, mutations.size());
        cache.release(lease);
        assertEquals(2, mutations.size(), "releasing references retains reusable keys");
        cache.evict(List.of(1L));
        assertEquals(new MockLruBlockCache.Mutation(3, 1L, false), mutations.get(2));
        cache.clear();
        assertNull(mutations.get(3).key());
        assertTrue(cache.keySnapshot().keys().isEmpty());
        assertEquals(4, cache.keySnapshot().version());
    }

    @Test void computedKeysAndAutomaticEvictionProduceTransitions() {
        var cache = new MockLruBlockCache(4);
        cache.setPrefixTreeEnabled(false);
        var mutations = new ArrayList<MockLruBlockCache.Mutation>();
        cache.setMutationListener(mutations::add);
        var lease = cache.acquire(1, List.of(7L));
        var retained = cache.retainComputed(lease, List.of(7L));
        assertEquals(List.of(new MockLruBlockCache.Mutation(1, 7L, true)), mutations);
        cache.release(retained);
        cache.admit(List.of(8L, 9L, 10L, 11L));
        assertTrue(mutations.stream().anyMatch(m -> !m.added() && Long.valueOf(7).equals(m.key())));
    }

    @Test void registersSnapshotsAddsDeletesHeartbeatsAndHostDown() throws Exception {
        var cache = new MockLruBlockCache(16);
        cache.admit(List.of(1L));
        var peer = new Peer();
        try (var publisher = new KvcmEventPublisher(config(64), cache, "host:61000", 128, () -> true, peer)) {
            publisher.start();
            until(() -> peer.keys.equals(Set.of(1L)) && publisher.status().get("state").equals("READY"));
            assertEquals("test-group", peer.requests.get(0).path("instance_group").asText());
            assertEquals(128, peer.requests.get(0).path("block_size").asInt());
            assertEquals("EVENT_NODE_REGISTER", peer.requests.get(1).path("events").path(0).path("event_type").asText());
            cache.admit(List.of(2L, Long.MIN_VALUE));
            until(() -> peer.keys.equals(Set.of(1L, 2L, Long.MIN_VALUE)));
            cache.evict(List.of(2L));
            until(() -> peer.keys.equals(Set.of(1L, Long.MIN_VALUE)));
            until(() -> !peer.events("EVENT_HEARTBEAT").isEmpty());
            cache.clear();
            until(() -> peer.keys.isEmpty() && peer.snapshots().size() >= 2);
        }
        assertFalse(peer.events("EVENT_BLOCK_ADD").isEmpty());
        assertFalse(peer.events("EVENT_BLOCK_DELETE").isEmpty());
        assertEquals(1, peer.events("EVENT_HOST_DOWN").size());
    }

    @Test void overflowNeverWaitsForNetworkAndResynchronizesWithoutStaleDeltas() throws Exception {
        var cache = new MockLruBlockCache(16);
        var peer = new Peer();
        peer.snapshotEntered = new CountDownLatch(1);
        peer.snapshotRelease = new CountDownLatch(1);
        try (var publisher = new KvcmEventPublisher(config(2), cache, "host:1", 128, () -> true, peer)) {
            publisher.start();
            assertTrue(peer.snapshotEntered.await(3, TimeUnit.SECONDS));
            assertTimeoutPreemptively(Duration.ofMillis(500), () -> {
                for (long i = 0; i < 100; i++) cache.admit(List.of(i));
            });
            assertTrue((long) publisher.status().get("overflows") > 0);
            peer.snapshotRelease.countDown();
            until(() -> peer.snapshots().size() >= 2 && publisher.status().get("state").equals("READY")
                    && peer.keys.equals(cache.keySnapshot().keys()));
            Thread.sleep(50);
            assertEquals(cache.keySnapshot().keys(), peer.keys);
        } finally { peer.snapshotRelease.countDown(); }
    }

    @Test void lostSnapshotAckRetriesIdenticalPayloadBeforeTakingNewSnapshot() throws Exception {
        var cache = new MockLruBlockCache(16);
        cache.admit(List.of(1L));
        var peer = new Peer();
        peer.loseSnapshotAck.set(true);
        try (var publisher = new KvcmEventPublisher(config(64), cache, "host:1", 128, () -> true, peer)) {
            publisher.start();
            until(() -> peer.snapshots().size() >= 3 && publisher.status().get("state").equals("READY"));
            assertEquals(peer.snapshots().get(0), peer.snapshots().get(1));
            assertTrue((long) publisher.status().get("failures") >= 1);
            assertEquals(Set.of(1L), peer.keys);
        }
    }

    @Test void lostMutationAckAndHeartbeatFailureRestoreAuthoritativeState() throws Exception {
        var cache = new MockLruBlockCache(16);
        var peer = new Peer();
        try (var publisher = new KvcmEventPublisher(config(64), cache, "host:1", 128, () -> true, peer)) {
            publisher.start();
            until(() -> publisher.status().get("state").equals("READY"));
            peer.loseMutationAck.set(true);
            cache.admit(List.of(9L));
            until(() -> peer.snapshots().size() >= 2 && peer.keys.equals(Set.of(9L)));
            peer.rejectHeartbeat.set(true);
            until(() -> (long) publisher.status().get("failures") >= 2 && peer.snapshots().size() >= 3);
            assertEquals(cache.keySnapshot().keys(), peer.keys);
        }
    }

    @Test void stoppedEngineStopsHeartbeatsAndRestartsWithSnapshot() throws Exception {
        var cache = new MockLruBlockCache(16);
        var running = new AtomicBoolean(true);
        var peer = new Peer();
        try (var publisher = new KvcmEventPublisher(config(64), cache, "host:1", 128, running::get, peer)) {
            publisher.start();
            until(() -> publisher.status().get("state").equals("READY"));
            running.set(false);
            until(() -> publisher.status().get("state").equals("STOPPED"));
            int count = peer.events("EVENT_HEARTBEAT").size();
            cache.admit(List.of(3L));
            Thread.sleep(80);
            assertEquals(count, peer.events("EVENT_HEARTBEAT").size());
            running.set(true);
            until(() -> peer.snapshots().size() >= 2 && peer.keys.equals(Set.of(3L)));
        }
    }

    @Test void coalescesDeleteThenAddOfSameKeyToFinalTransition() throws Exception {
        var cache = new MockLruBlockCache(8);
        var peer = new Peer();
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        var firstHeartbeat = new AtomicBoolean(true);
        KvcmEventPublisher.Transport transport = (route, body) -> {
            if (body.contains("EVENT_HEARTBEAT") && firstHeartbeat.getAndSet(false)) {
                entered.countDown();
                assertTrue(release.await(3, TimeUnit.SECONDS));
            }
            return peer.post(route, body);
        };
        try (var publisher = new KvcmEventPublisher(config(64), cache, "host:1", 128, () -> true, transport)) {
            publisher.start();
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            cache.admit(List.of(5L));
            cache.evict(List.of(5L));
            cache.admit(List.of(5L));
            release.countDown();
            until(() -> peer.keys.equals(Set.of(5L)));
            assertEquals(1, peer.events("EVENT_BLOCK_ADD").size());
            assertEquals(0, peer.events("EVENT_BLOCK_DELETE").size());
        } finally { release.countDown(); }
    }

    @Test void periodicSnapshotRepairsServerStateLostWithoutRequestFailure() throws Exception {
        var cache = new MockLruBlockCache(8);
        cache.admit(List.of(8L));
        var peer = new Peer();
        var base = config(64);
        var periodic = new KvcmEventPublisher.Config(base.endpoint(), base.group(), base.instance(), base.model(),
                64, 16, 5, 30, 80, 500, 5, 50);
        try (var publisher = new KvcmEventPublisher(periodic, cache, "host:1", 128, () -> true, peer)) {
            publisher.start();
            until(() -> publisher.status().get("state").equals("READY"));
            peer.keys.clear();
            until(() -> peer.snapshots().size() >= 2 && peer.keys.equals(Set.of(8L)));
        }
    }

    @Test void httpTransportChecksApplicationStatusAndUsesUniqueEngineIdentities() throws Exception {
        var requests = new CopyOnWriteArrayList<JsonNode>();
        var reject = new AtomicInteger(1);
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/api/", exchange -> {
            requests.add(JSON.readTree(exchange.getRequestBody()));
            byte[] response = (reject.getAndDecrement() > 0 ? "{}" : OK).getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            try (var out = exchange.getResponseBody()) { out.write(response); }
            exchange.close();
        });
        server.start();
        var cfg = config(URI.create("http://127.0.0.1:" + server.getAddress().getPort()), 64);
        try (var first = new KvcmEventPublisher(cfg, new MockLruBlockCache(8), "host:1", 128, () -> true, null);
             var second = new KvcmEventPublisher(cfg, new MockLruBlockCache(8), "host:2", 16, () -> true, null)) {
            first.start(); second.start();
            until(() -> first.status().get("state").equals("READY") && second.status().get("state").equals("READY"));
            Set<String> instances = new HashSet<>();
            requests.stream().filter(root -> root.has("instance_group")).forEach(root -> instances.add(root.path("instance_id").asText()));
            assertEquals(Set.of("test-deployment_128", "test-deployment_16"), instances);
            assertEquals(1L, (long) first.status().get("failures") + (long) second.status().get("failures"));
        } finally { server.stop(0); }
    }

    @Test void sameBlockSizeSharesQueryNamespaceButKeepsDistinctOwners() throws Exception {
        var cfg = KvcmEventPublisher.Config.fromEnv(Map.of("KV_CACHE_EVENT_PUBLISHER_TYPE", "kvcm",
                "KV_CACHE_EVENT_MANAGER_ENDPOINT", "http://localhost:1", "KV_CACHE_EVENT_INSTANCE_GROUP", "test-group",
                "KV_CACHE_EVENT_INSTANCE_ID", "model-deployment", "MOCK_KVCM_STORAGE_TYPE", "ST_EVENT_REPORT_L1P5",
                "MOCK_KVCM_FLUSH_MS", "5"));
        var firstPeer = new Peer(); var secondPeer = new Peer();
        var firstCache = new MockLruBlockCache(8); var secondCache = new MockLruBlockCache(8);
        firstCache.admit(List.of(42L)); secondCache.admit(List.of(42L));
        try (var first = new KvcmEventPublisher(cfg, firstCache, "host:1", 128, () -> true, firstPeer);
             var second = new KvcmEventPublisher(cfg, secondCache, "host:2", 128, () -> true, secondPeer)) {
            first.start(); second.start();
            until(() -> !firstPeer.snapshots().isEmpty() && !secondPeer.snapshots().isEmpty());
            for (var peer : List.of(firstPeer, secondPeer)) {
                JsonNode snapshot = peer.snapshots().getFirst();
                assertEquals("model-deployment_128", snapshot.path("instance_id").asText());
                assertEquals("ST_EVENT_REPORT_L1P5", snapshot.path("storage_type").asText());
            }
            assertEquals("host:1", firstPeer.snapshots().getFirst().path("host_ip_port").asText());
            assertEquals("host:2", secondPeer.snapshots().getFirst().path("host_ip_port").asText());
            firstCache.evict(List.of(42L));
            until(() -> firstPeer.keys.isEmpty());
            assertEquals(Set.of(42L), secondPeer.keys);
        }
    }

    @Test void closeRetriesHostDownInterruptedByEngineStop() throws Exception {
        var peer = new Peer();
        var available = new AtomicBoolean(true);
        var entered = new CountDownLatch(1);
        var attempts = new AtomicInteger();
        KvcmEventPublisher.Transport transport = (route, body) -> {
            if (JSON.readTree(body).path("events").path(0).path("event_type").asText().equals("EVENT_HOST_DOWN")
                    && attempts.incrementAndGet() == 1) {
                entered.countDown();
                new CountDownLatch(1).await(3, TimeUnit.SECONDS);
            }
            return peer.post(route, body);
        };
        try (var publisher = new KvcmEventPublisher(config(16), new MockLruBlockCache(8), "host:1", 128,
                available::get, transport)) {
            publisher.start();
            until(() -> publisher.status().get("state").equals("READY"));
            available.set(false);
            assertTrue(entered.await(3, TimeUnit.SECONDS));
            publisher.close();
            assertEquals(2, attempts.get());
            assertEquals(1, peer.events("EVENT_HOST_DOWN").size());
            assertEquals("CLOSED", publisher.status().get("state"));
        }
    }
}
