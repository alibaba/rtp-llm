package org.flexlb.mock.protocol;

import static org.junit.jupiter.api.Assertions.*;
import java.net.*;
import java.net.http.*;
import java.util.*;
import org.junit.jupiter.api.*;

/**
 * Verifies control request validation and error responses.
 */
class ControlHttpServerTest {
    private static final HttpClient CLIENT = HttpClient.newHttpClient();

    private ControlHttpServer server;

    @BeforeEach void start() throws Exception {
        server = TransportProbe.server();
    }

    @AfterEach void close() { server.close(); }

    @Test void optionalMetricsEndpointIsExactAndReportsCollectionFailure() throws Exception {
        var fail = new java.util.concurrent.atomic.AtomicBoolean();
        try (var exporter = new ControlHttpServer(new InetSocketAddress("127.0.0.1", 0)).metrics(() -> {
            if (fail.get()) throw new IllegalStateException("broken collector");
            return "# TYPE probe_total counter\nprobe_total 3\n";
        }).start()) {
            var request = HttpRequest.newBuilder(URI.create(exporter.url() + "/metrics")).GET().build();
            var response = CLIENT.send(request, HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode()); assertTrue(response.body().contains("probe_total 3"));
            assertEquals("text/plain; version=0.0.4; charset=utf-8", response.headers().firstValue("Content-Type").orElseThrow());
            for (String path : List.of("/metrics/unknown", "/metrics?unexpected=1"))
                assertEquals(404, CLIENT.send(HttpRequest.newBuilder(URI.create(exporter.url() + path)).GET().build(), HttpResponse.BodyHandlers.ofString()).statusCode());
            assertEquals(404, CLIENT.send(HttpRequest.newBuilder(request.uri()).POST(HttpRequest.BodyPublishers.noBody()).build(), HttpResponse.BodyHandlers.ofString()).statusCode());
            fail.set(true);
            assertEquals(500, CLIENT.send(request, HttpResponse.BodyHandlers.ofString()).statusCode());
        }
        assertEquals(404, CLIENT.send(HttpRequest.newBuilder(URI.create(server.url() + "/metrics")).GET().build(), HttpResponse.BodyHandlers.ofString()).statusCode());
    }

    @Test void typedRecords() throws Exception {
        var response = call("probe/action", Map.of(), 1000);
        assertEquals(200, response.statusCode());
        assertEquals("master-0", ControlJson.JSON.readTree(response.body()).at("/result/affected/instances/0/id").asText());
    }

    @Test void unhealthyComponentsReturnUnavailableWithoutCollectingMetrics() throws Exception {
        var broken = new java.util.concurrent.atomic.AtomicBoolean();
        var scrapes = new java.util.concurrent.atomic.AtomicInteger();
        try (var service = new ControlHttpServer(new InetSocketAddress("127.0.0.1", 0))
                .health(() -> { if (broken.get()) throw new IllegalStateException("worker failed"); })
                .metrics(() -> { scrapes.incrementAndGet(); return ""; }).start()) {
            var request = HttpRequest.newBuilder(URI.create(service.url() + "/health")).GET().build();
            assertEquals(200, CLIENT.send(request, HttpResponse.BodyHandlers.ofString()).statusCode());
            broken.set(true);
            assertEquals(503, CLIENT.send(request, HttpResponse.BodyHandlers.ofString()).statusCode());
            assertEquals(0, scrapes.get());
        }
    }

    @Test void nativeOnlyNodeIsHealthyWithoutInvokeRoute() throws Exception {
        try (var nativeHost = new ControlHttpServer(new InetSocketAddress("127.0.0.1", 0)).start()) {
            var response = CLIENT.send(HttpRequest.newBuilder(URI.create(nativeHost.url() + "/health")).GET().build(), HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode());
            assertEquals(ControlJson.JSON.readTree("{\"status\":\"ok\"}"), ControlJson.JSON.readTree(response.body()));
            var invoke = HttpRequest.newBuilder(URI.create(nativeHost.url() + "/invoke"))
                .POST(HttpRequest.BodyPublishers.noBody()).build();
            assertEquals(404, CLIENT.send(invoke, HttpResponse.BodyHandlers.ofString()).statusCode());
        }
    }

    @Test void rejectsInvalidRequestsBeforeImplementation() throws Exception {
        assertEquals(400, call("probe/action", Map.of("delay_ms", java.math.BigInteger.ONE.shiftLeft(63)), 1000).statusCode());
        assertEquals(400, call("probe/action", Map.of("delay_ms", "1"), 1000).statusCode());
        assertEquals(400, call("probe/action", Map.of("extra", 1), 1000).statusCode());
        assertEquals(400, call("probe/missing", Map.of(), 1000).statusCode());
    }

    @Test void timesOutWithoutRetry() throws Exception {
        assertEquals(504, call("probe/action", Map.of("delay_ms", 1000), 20).statusCode());
        assertEquals(200, call("probe/metric", Map.of(), 1000).statusCode());
    }

    @Test void rejectsInvalidComponentResult() throws Exception {
        assertEquals(500, call("probe/metric", Map.of("invalid_result", true), 1000).statusCode());
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"127.0.0.1", "127.0.0.2", "::1"})
    void publishesTheActualBoundAddress(String host) throws Exception {
        ControlHttpServer bound;
        try { bound = new ControlHttpServer(new InetSocketAddress(host, 0)); }
        catch (java.net.SocketException unavailable) {
            if (unavailable instanceof java.net.BindException || unavailable.getCause() instanceof java.nio.channels.UnsupportedAddressTypeException) {
                org.junit.jupiter.api.Assumptions.assumeTrue(false, "Loopback address unavailable: " + host); return;
            }
            throw unavailable;
        }
        try (bound) {
            bound.start();
            String published = URI.create(bound.url()).getHost().replace("[", "").replace("]", "");
            assertEquals(java.net.InetAddress.getByName(host), java.net.InetAddress.getByName(published));
            assertEquals(200, CLIENT.send(HttpRequest.newBuilder(URI.create(bound.url() + "/health")).GET().build(),
                HttpResponse.BodyHandlers.ofString()).statusCode());
        }
    }

    private HttpResponse<String> call(String route, Object args, long timeout) throws Exception {
        var body = ControlJson.JSON.writeValueAsString(Map.of("interface", route, "args", args, "timeout_ms", timeout));
        return CLIENT.send(HttpRequest.newBuilder(URI.create(server.url() + "/invoke"))
            .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
    }

}
