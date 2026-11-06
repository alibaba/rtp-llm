package org.flexlb.mock.engine.cache;

import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.HttpServer;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.BooleanSupplier;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class KvcmReporterTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static void await(BooleanSupplier condition) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(8);
        while (!condition.getAsBoolean()) {
            assertTrue(System.nanoTime() < deadline, "KVCM state timed out"); Thread.sleep(10);
        }
    }
    @ParameterizedTest @ValueSource(booleans = {false, true})
    void slowFailedSnapshotDoesNotBlockDeltasOrReplayAcrossRegistration(boolean registrationLost) throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var heartbeats = new AtomicInteger(); var registrations = new AtomicInteger();
        var expire = new AtomicBoolean(registrationLost); var hint = new AtomicBoolean();
        var bodies = new CopyOnWriteArrayList<String>(); var added = new AtomicBoolean();
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        try (var threads = Executors.newVirtualThreadPerTaskExecutor(); var cache = new KvCacheManager(1, 8, 0)) {
            server.setExecutor(threads);
            server.createContext("/api/", exchange -> {
                try (exchange) {
                    String raw = new String(exchange.getRequestBody().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
                    var body = JSON.readTree(raw);
                    String type = body.path("events").path(0).path("event_type").asText();
                    var response = JSON.createObjectNode();
                    response.putObject("header").putObject("status").put("code", "OK");
                    if (body.has("model_deployment")) {
                        registrations.incrementAndGet();
                        response.put("opaque_info", "x".repeat(1048577));
                    }
                    if (type.equals("EVENT_HEARTBEAT")) {
                        heartbeats.incrementAndGet();
                        if (entered.getCount() == 0 && expire.compareAndSet(true, false))
                            response.with("header").with("status").put("code", "INSTANCE_NOT_EXIST");
                        response.put("snapshot_required", hint.getAndSet(false));
                    }
                    if (type.equals("EVENT_BLOCK_ADD")) added.set(true);
                    if (type.equals("EVENT_BLOCK_SNAPSHOT")) {
                        bodies.add(raw);
                        if (bodies.size() == 1) {
                            entered.countDown(); release.await();
                            response.with("header").with("status").put("code", registrationLost ? "INSTANCE_NOT_EXIST" : "SNAPSHOT_RATE_LIMITED");
                            response.put("retry_after_ms", "18446744073709551615");
                        }
                    }
                    byte[] bytes = JSON.writeValueAsBytes(response);
                    exchange.sendResponseHeaders(200, bytes.length); exchange.getResponseBody().write(bytes);
                } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            });
            server.start();
            try (var reporter = new KvcmReporter(new KvcmReporter.Configuration(
                    URI.create("http://127.0.0.1:" + server.getAddress().getPort()), true, 600000),
                    new KvcmReporter.Identity("instance", "group", "127.0.0.1:9000", "model", "FP16", false, 1, 1, 1, "full"), cache)) {
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                int before = heartbeats.get(); await(() -> heartbeats.get() > before);
                if (registrationLost) {
                    await(() -> Boolean.FALSE.equals(reporter.status().get("registered")));
                    assertEquals(1, registrations.get(), "Re-registration must drain an old snapshot upload");
                }
                try (var lease = cache.acquire(1, List.of()).lease()) { lease.publish(List.of(99L), 1); }
                if (!registrationLost) await(added::get); // The first snapshot is still uploading.
                release.countDown();
                if (!registrationLost) {
                    await(() -> reporter.status().get("last_error").toString().contains("SNAPSHOT_RATE_LIMITED"));
                    int beats = heartbeats.get(); await(() -> heartbeats.get() > beats);
                    assertEquals(1, bodies.size(), "Failed snapshots must not be replayed");
                    hint.set(true);
                }
                await(() -> bodies.size() == 2);
                var blocks = JSON.readTree(bodies.get(1)).path("events").get(0).path("block_snapshot").path("blocks");
                assertEquals(1, blocks.size()); assertEquals("99", blocks.get(0).path("block_key").asText());
                assertNotEquals(bodies.get(0), bodies.get(1));
                await(() -> reporter.status().get("last_error").equals(""));
                assertEquals(registrationLost ? 2 : 1, registrations.get(), "Stale snapshot feedback must not invalidate a new registration");
            } finally { release.countDown(); server.stop(0); }
        }
    }
    private static final class Meta implements AutoCloseable {
        final HttpServer server;
        final ExecutorService threads = Executors.newVirtualThreadPerTaskExecutor();
        final List<JsonNode> requests = new CopyOnWriteArrayList<>();
        volatile java.util.function.BiConsumer<JsonNode, com.fasterxml.jackson.databind.node.ObjectNode> reply = (request, response) -> {};
        Meta() throws Exception {
            server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
            server.setExecutor(threads);
            server.createContext("/api/", exchange -> {
                try (exchange) {
                    var request = JSON.readTree(exchange.getRequestBody()); requests.add(request);
                    var response = JSON.createObjectNode(); response.putObject("header").putObject("status").put("code", "OK");
                    reply.accept(request, response);
                    byte[] bytes = JSON.writeValueAsBytes(response);
                    exchange.sendResponseHeaders(200, bytes.length); exchange.getResponseBody().write(bytes);
                }
            });
            server.start();
        }
        int count(String type) { return (int) requests.stream().filter(request -> type.equals(type(request))).count(); }
        int registrations() { return (int) requests.stream().filter(request -> request.has("model_deployment")).count(); }
        static String type(JsonNode request) { return request.path("events").path(0).path("event_type").asText(); }
        KvcmReporter reporter(KvCacheManager cache, boolean events, java.util.function.LongSupplier clock) {
            return new KvcmReporter(new KvcmReporter.Configuration(
                URI.create("http://127.0.0.1:" + server.getAddress().getPort()), events, 600000),
                new KvcmReporter.Identity("instance", "group", "127.0.0.1:9000", "model", "FP16", false, 1, 1, 1, "full"), cache, clock);
        }
        @Override public void close() { server.stop(0); threads.close(); }
    }
    private static void heartbeat(Meta meta) throws Exception {
        int before = meta.count("EVENT_HEARTBEAT");
        // Two starts guarantee the preceding response has been processed by the serial heartbeat worker.
        await(() -> meta.count("EVENT_HEARTBEAT") >= before + 2);
    }

    @ParameterizedTest @ValueSource(booleans = {false, true})
    void allItemResultsAreScannedEvenAfterHeaderOrEarlierItemFailure(boolean headerFailure) throws Exception {
        try (var meta = new Meta(); var cache = new KvCacheManager(1, 8, 0)) {
            var reject = new AtomicBoolean(true);
            meta.reply = (request, response) -> {
                if (Meta.type(request).equals("EVENT_HEARTBEAT") && reject.compareAndSet(true, false)) {
                    response.withObject("/header/status").put("code", headerFailure ? "ERROR" : "OK");
                    response.putArray("item_results").addObject().put("code", "ERROR");
                    response.withArray("item_results").addObject().put("code", "INSTANCE_NOT_EXIST");
                }
            };
            try (var reporter = meta.reporter(cache, true, System::nanoTime)) {
                await(() -> meta.registrations() == 2 && Boolean.TRUE.equals(reporter.status().get("registered")));
                heartbeat(meta); assertEquals(2, meta.registrations());
            }
        }
    }

    @Test void repeatedHeartbeatHintsTriggerOneSnapshotAndFalseRearmsTheHint() throws Exception {
        try (var meta = new Meta(); var cache = new KvCacheManager(1, 8, 0)) {
            var hint = new AtomicBoolean(false); var expire = new AtomicBoolean();
            meta.reply = (request, response) -> {
                if (Meta.type(request).equals("EVENT_HEARTBEAT")) {
                    response.put("snapshot_required", hint.get());
                    if (expire.compareAndSet(true, false)) response.withObject("/header/status").put("code", "INSTANCE_NOT_EXIST");
                }
                // Snapshot ACKs do not themselves request another snapshot.
                if (Meta.type(request).equals("EVENT_BLOCK_SNAPSHOT")) response.put("snapshot_required", true);
            };
            try (var reporter = meta.reporter(cache, true, System::nanoTime)) {
                await(() -> meta.count("EVENT_BLOCK_SNAPSHOT") == 1);
                heartbeat(meta); hint.set(true);
                await(() -> meta.count("EVENT_BLOCK_SNAPSHOT") == 2);
                heartbeat(meta); assertEquals(2, meta.count("EVENT_BLOCK_SNAPSHOT"));
                hint.set(false); heartbeat(meta);
                hint.set(true); await(() -> meta.count("EVENT_BLOCK_SNAPSHOT") == 3);
                heartbeat(meta); assertEquals(3, meta.count("EVENT_BLOCK_SNAPSHOT"));
                expire.set(true); await(() -> meta.registrations() == 2);
                await(() -> meta.count("EVENT_BLOCK_SNAPSHOT") == 5);
                heartbeat(meta); assertEquals(5, meta.count("EVENT_BLOCK_SNAPSHOT"), "Re-registration rearms the hint exactly once");
            }
        }
    }

    @Test void missingNormalAckRecoversAtSixtySecondsOnceAndRegistrationDoesNotRearmIt() throws Exception {
        try (var meta = new Meta(); var cache = new KvCacheManager(1, 8, 0)) {
            var clock = new AtomicLong(1); var fail = new AtomicBoolean(true);
            meta.reply = (request, response) -> {
                if (Meta.type(request).equals("EVENT_HEARTBEAT") && fail.get())
                    response.withObject("/header/status").put("code", "ERROR");
            };
            try (var reporter = meta.reporter(cache, true, clock::get)) {
                await(() -> meta.count("EVENT_BLOCK_SNAPSHOT") == 1);
                heartbeat(meta); clock.addAndGet(TimeUnit.SECONDS.toNanos(120));
                heartbeat(meta); assertEquals(1, meta.registrations(), "Registration and snapshot ACKs do not arm recovery");
                fail.set(false); await(() -> reporter.status().get("last_error").equals(""));
                fail.set(true); clock.addAndGet(TimeUnit.SECONDS.toNanos(59));
                await(() -> !reporter.status().get("last_error").equals(""));
                assertEquals(1, meta.registrations());
                clock.addAndGet(TimeUnit.SECONDS.toNanos(1));
                await(() -> meta.registrations() == 2 && Boolean.TRUE.equals(reporter.status().get("registered")));
                clock.addAndGet(TimeUnit.SECONDS.toNanos(120));
                heartbeat(meta); assertEquals(2, meta.registrations(), "No repeated registration without a new normal ACK");
            }
        }
    }

    @Test void aLaterMutationAckRefreshesTheRecoveryWindowWhileHeartbeatsFail() throws Exception {
        var releaseHeartbeat = new CountDownLatch(1);
        try (var meta = new Meta(); var cache = new KvCacheManager(1, 8, 0)) {
            var now = new AtomicLong(1);
            var observed = new AtomicReference<>(new CountDownLatch(1));
            meta.reply = (request, response) -> {
                if (Meta.type(request).equals("EVENT_HEARTBEAT")) {
                    try { releaseHeartbeat.await(); }
                    catch (InterruptedException error) { Thread.currentThread().interrupt(); }
                    response.withObject("/header/status").put("code", "ERROR");
                }
            };
            try (var reporter = meta.reporter(cache, true, () -> {
                long value = now.get(); observed.get().countDown(); return value;
            })) {
                try {
                    await(() -> meta.count("EVENT_BLOCK_SNAPSHOT") == 1);
                    try (var lease = cache.acquire(1, List.of()).lease()) { lease.publish(List.of(11L), 1); }
                    assertTrue(observed.get().await(2, TimeUnit.SECONDS));
                    now.addAndGet(TimeUnit.SECONDS.toNanos(59)); observed.set(new CountDownLatch(1));
                    try (var lease = cache.acquire(1, List.of()).lease()) { lease.publish(List.of(22L), 1); }
                    assertTrue(observed.get().await(2, TimeUnit.SECONDS));
                    now.addAndGet(TimeUnit.SECONDS.toNanos(1)); releaseHeartbeat.countDown();
                    heartbeat(meta); assertEquals(1, meta.registrations(), "The mutation ACK refreshed the original deadline");
                    now.addAndGet(TimeUnit.SECONDS.toNanos(59));
                    await(() -> meta.registrations() == 2);
                } finally { releaseHeartbeat.countDown(); }
            }
        }
    }

}
