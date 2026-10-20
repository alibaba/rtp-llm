package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.Test;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BooleanSupplier;
import static org.junit.jupiter.api.Assertions.*;

class KvcmEventPublisherTest {
    private static final ObjectMapper JSON = new ObjectMapper();

    private void await(BooleanSupplier predicate) throws Exception {
        long end = System.nanoTime() + Duration.ofSeconds(12).toNanos();
        while (!predicate.getAsBoolean() && System.nanoTime() < end) Thread.sleep(10);
        assertTrue(predicate.getAsBoolean(), "publisher did not converge");
    }

    @Test void lifecycleMutationsResetAndRecovery() throws Exception {
        var bodies = new CopyOnWriteArrayList<JsonNode>();
        var fail = new AtomicBoolean(false);
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/", exchange -> {
            bodies.add(JSON.readTree(exchange.getRequestBody()));
            byte[] response = (fail.get() ? "{\"header\":{\"status\":{\"code\":\"INVALID_ARGUMENT\"}}}"
                    : "{\"header\":{\"status\":{\"code\":\"OK\"}}}").getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, response.length);
            exchange.getResponseBody().write(response);
            exchange.close();
        });
        server.start();
        var cache = new MockLruBlockCache(16);
        try (var pub = new KvcmEventPublisher(cache, "http://127.0.0.1:" + server.getAddress().getPort(),
                "group", "instance", "127.0.0.1:1", 128, "ST_EVENT_REPORT_L1P5", 4)) {
            await(() -> (long) pub.status().get("snapshots") >= 1);
            cache.admit(List.of(1L, 2L));
            await(() -> bodies.stream().anyMatch(b -> b.toString().contains("EVENT_BLOCK_ADD")));
            long adds = bodies.stream().filter(b -> b.toString().contains("EVENT_BLOCK_ADD")).count();
            cache.admit(List.of(1L, 2L));
            Thread.sleep(150);
            assertEquals(adds, bodies.stream().filter(b -> b.toString().contains("EVENT_BLOCK_ADD")).count());
            cache.evict(List.of(1L));
            await(() -> bodies.stream().anyMatch(b -> b.toString().contains("EVENT_BLOCK_DELETE")));
            fail.set(true);
            cache.admit(List.of(3L));
            await(() -> (long) pub.status().get("failures") >= 1);
            // An unreachable/erroring endpoint cannot hold the allocation lock.
            long start = System.nanoTime();
            for (long key = 10; key < 100; key++) cache.admit(List.of(key));
            assertTrue(System.nanoTime() - start < Duration.ofSeconds(1).toNanos());
            fail.set(false);
            await(() -> (long) pub.status().get("snapshots") >= 2);
            long snapshots = (long) pub.status().get("snapshots");
            cache.clear();
            await(() -> (long) pub.status().get("snapshots") > snapshots);
            assertTrue(bodies.stream().anyMatch(b -> b.toString().contains("\"blocks\":[]")));
        } finally { server.stop(0); }
        assertTrue(bodies.stream().anyMatch(b -> b.toString().contains("EVENT_HOST_DOWN")));
    }

    @Test void checksBusinessAndItemStatuses() throws Exception {
        assertTrue(KvcmEventPublisher.ok(JSON.readTree("{\"status\":{\"code\":1}}")));
        assertTrue(KvcmEventPublisher.ok(JSON.readTree("\"OK\"")));
        assertFalse(KvcmEventPublisher.ok(JSON.readTree("{\"code\":\"INVALID_ARGUMENT\"}")));
        assertFalse(KvcmEventPublisher.ok(JSON.readTree("{}")));
    }
}
