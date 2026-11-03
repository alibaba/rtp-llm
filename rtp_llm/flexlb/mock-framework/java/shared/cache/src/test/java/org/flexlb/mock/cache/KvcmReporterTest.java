package org.flexlb.mock.cache;

import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.HttpServer;
import java.net.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.BooleanSupplier;
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
        try (var threads = Executors.newVirtualThreadPerTaskExecutor(); var cache = new KvCache(1, 8, 0, KvCache.EvictionPolicy.LRU)) {
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
}
