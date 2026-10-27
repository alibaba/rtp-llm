package org.flexlb.mock.metrics;

import static org.junit.jupiter.api.Assertions.*;
import com.fasterxml.jackson.databind.*;
import com.sun.net.httpserver.HttpServer;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import org.flexlb.mock.api.engine.EngineMetricsApi;
import org.flexlb.mock.api.metrics.PrometheusApi;
import org.flexlb.mock.runtime.ApiServer;
import org.junit.jupiter.api.*;

class PrometheusQueriesTest {
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Instant TIME = Instant.parse("2026-01-02T03:04:05.123456Z");
    private final HttpClient client = HttpClient.newHttpClient();
    private final ExecutorService handlers = Executors.newVirtualThreadPerTaskExecutor();
    private final AtomicReference<String> response = new AtomicReference<>("{\"status\":\"success\",\"data\":{\"resultType\":\"scalar\",\"result\":[1767323045.123,\"0.4\"]}}");
    private final AtomicReference<Map<String, String>> params = new AtomicReference<>();
    private final AtomicInteger calls = new AtomicInteger(), status = new AtomicInteger(200);
    private final AtomicLong delay = new AtomicLong();
    private HttpServer backend;
    private PrometheusQueries queries;
    private ApiServer api;

    @BeforeEach void start() throws Exception {
        backend = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0); backend.setExecutor(handlers);
        backend.createContext("/prefix/api/v1/query", exchange -> {
            try (exchange) {
                var form = new HashMap<String, String>();
                for (String item : new String(exchange.getRequestBody().readAllBytes(), StandardCharsets.UTF_8).split("&")) {
                    var pair = item.split("=", 2); form.put(URLDecoder.decode(pair[0], StandardCharsets.UTF_8), URLDecoder.decode(pair[1], StandardCharsets.UTF_8));
                }
                params.set(form); calls.incrementAndGet();
                try { Thread.sleep(delay.get()); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
                byte[] body = response.get().getBytes(StandardCharsets.UTF_8);
                exchange.sendResponseHeaders(status.get(), body.length); exchange.getResponseBody().write(body);
            }
        }); backend.start();
        queries = new PrometheusQueries(URI.create("http://127.0.0.1:" + backend.getAddress().getPort() + "/prefix/"), "experiment", "rtp");
        api = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).bind(EngineMetricsApi.class, queries).bind(PrometheusApi.class, queries).start();
    }
    @AfterEach void close() {
        api.close(); queries.close(); backend.stop(0); handlers.shutdownNow(); client.close();
    }
    private HttpResponse<String> invoke(String route, Map<String, Object> args, Object observed, long budget) throws Exception {
        var context = new HashMap<String, Object>();
        context.put("run", "foreground"); context.put("query_time", TIME.toString());
        context.put("observation", observed); context.put("timeout_ms", budget);
        var payload = Map.of("interface", route, "args", args, "context", context);
        return client.send(HttpRequest.newBuilder(URI.create(api.url() + "/invoke")).timeout(java.time.Duration.ofSeconds(5))
            .POST(HttpRequest.BodyPublishers.ofString(JSON.writeValueAsString(payload))).build(), HttpResponse.BodyHandlers.ofString());
    }
    private Map<String, Object> interval() {
        double at = TIME.getEpochSecond() + TIME.getNano() / 1e9;
        return Map.of("selector", List.of("-60s", "-10s"), "start", at - 60, "end", at - 10);
    }

    @Test void rawPromqlUsesFixedTimeAndPreservesVectorLabelsAndSampleTime() throws Exception {
        String expression = "max_over_time(example{kind=\"a+b & 中文\"}[1m])";
        response.set("{\"status\":\"success\",\"data\":{\"resultType\":\"vector\",\"result\":[{\"metric\":{\"node\":\"p-0\"},\"value\":[1767323045.123,\"2\"]}]}}");
        var result = invoke("prometheus/query", Map.of("promql", expression), null, 2000);
        assertEquals(200, result.statusCode(), result.body());
        var actual = JSON.readTree(result.body()).get("result");
        assertEquals(TIME.toString(), actual.get("query_time").asText());
        assertEquals("p-0", actual.path("samples").get(0).path("labels").get("node").asText());
        assertEquals(2, actual.path("samples").get(0).get("value").asDouble());
        assertEquals(expression, params.get().get("query")); assertEquals(TIME.toString(), params.get().get("time"));
        assertEquals("2000ms", params.get().get("timeout"));
        assertEquals(1, calls.get());
    }

    @Test void intervalUsesResolvedEndAndTokenWeightedPromqlWithEscapedSelection() throws Exception {
        var args = Map.<String, Object>of("range", List.of("-60s", "-10s"), "engines", Map.of("instances", List.of(Map.of("id", "p.1"), Map.of("id", "p|2"))));
        var result = invoke("engine/cache/hit_rate", args, interval(), 2000);
        assertEquals(200, result.statusCode(), result.body());
        assertEquals(.4, JSON.readTree(result.body()).path("result").path("value").asDouble());
        String query = params.get().get("query");
        assertTrue(query.contains("sum(increase(" + EngineMetrics.CACHE_HITS));
        assertTrue(query.contains(" / sum(increase(" + EngineMetrics.CACHE_LOOKUPS));
        assertTrue(query.contains("[50000ms]")); assertTrue(query.contains("job=\"experiment\",engine=\"rtp\",role=\"prefill\""));
        assertTrue(query.contains("node=~\"p\\\\.1|p\\\\|2\""), query);
        assertEquals(TIME.minusSeconds(10).truncatedTo(java.time.temporal.ChronoUnit.MILLIS).toString(), params.get().get("time"));
    }

    @Test void latestUsesLastScrapeIntervalAndComputeTpsDividesByForwardTime() throws Exception {
        var observed = Map.of("selector", "latest", "at", "2026-01-02T03:04:05.123456+00:00");
        for (String path : List.of("engine/prefill/tps", "engine/prefill/tps_with_cache", "engine/decode/tps")) {
            var result = invoke(path, Map.of("range", "latest"), observed, 2000);
            assertEquals(200, result.statusCode(), result.body());
            String query = params.get().get("query");
            assertTrue(query.contains("irate(")); assertTrue(query.contains("[60000ms]"));
            if (path.contains("prefill")) assertTrue(query.contains(" / sum(irate(" + EngineMetrics.FORWARD_SECONDS + "_sum"));
            else assertFalse(query.contains(" / "));
        }
        invoke("engine/decode/tps", Map.of("range", List.of("-60s", "-10s")), interval(), 2000);
        assertTrue(params.get().get("query").endsWith(" / 50.000"));
    }

    @Test void rejectsNonfinitePartialMalformedAndUnsupportedBackendResults() throws Exception {
        for (String data : List.of(
                "{\"resultType\":\"scalar\",\"result\":[1,\"NaN\"]}",
                "{\"resultType\":\"scalar\",\"result\":[1,\"+Inf\"]}",
                "{\"resultType\":\"matrix\",\"result\":[]}",
                "{\"resultType\":\"string\",\"result\":[1,\"text\"]}",
                "{\"resultType\":\"vector\",\"result\":[{\"metric\":{},\"histogram\":[1,{}]}]}")) {
            response.set("{\"status\":\"success\",\"data\":" + data + "}");
            assertEquals(500, invoke("prometheus/query", Map.of("promql", "x"), null, 2000).statusCode(), data);
        }
        response.set("{\"status\":\"success\",\"warnings\":[\"partial data\"],\"data\":{\"resultType\":\"scalar\",\"result\":[1,\"1\"]}}");
        assertEquals(500, invoke("prometheus/query", Map.of("promql", "x"), null, 2000).statusCode());
        response.set("{broken");
        assertEquals(500, invoke("prometheus/query", Map.of("promql", "x"), null, 2000).statusCode());
        status.set(503);
        assertEquals(500, invoke("prometheus/query", Map.of("promql", "x"), null, 2000).statusCode());
    }

    @Test void noObservationsCannotBecomeASuccessfulSemanticMetric() throws Exception {
        response.set("{\"status\":\"success\",\"data\":{\"resultType\":\"vector\",\"result\":[]}}");
        assertEquals(200, invoke("prometheus/query", Map.of("promql", "absent_metric"), null, 2000).statusCode());
        assertEquals(500, invoke("engine/cache/hit_rate", Map.of("range", List.of("-60s", "-10s")), interval(), 2000).statusCode());
        assertEquals(500, invoke("engine/cache/hit_rate", Map.of("range", List.of("-60s", "-10s")), null, 2000).statusCode());
        assertEquals(500, invoke("prometheus/query", Map.of("promql", "x"), interval(), 2000).statusCode());
        assertEquals(500, invoke("engine/cache/hit_rate", Map.of("range", "latest", "engines", Map.of("instances", List.of())), Map.of("selector", "latest", "at", TIME.toString()), 2000).statusCode());
    }

    @Test void queryTimeoutDoesNotRetryOrLeakThePendingCall() throws Exception {
        assertEquals(200, invoke("prometheus/query", Map.of("promql", "warmup"), null, 2000).statusCode());
        calls.set(0); delay.set(2000);
        int result = invoke("prometheus/query", Map.of("promql", "slow"), null, 500).statusCode();
        assertTrue(result == 500 || result == 504); assertEquals(1, calls.get());
    }

    @Test void readOnlyBindingDoesNotAdvertiseScaleInAndInvalidConfigIsRejected() throws Exception {
        var health = client.send(HttpRequest.newBuilder(URI.create(api.url() + "/health")).GET().build(), HttpResponse.BodyHandlers.ofString());
        assertFalse(health.body().contains("scale_in"));
        assertThrows(IllegalArgumentException.class, () -> new PrometheusQueries(URI.create("file:///tmp/x"), "x", "rtp"));
        assertThrows(IllegalArgumentException.class, () -> PrometheusQueries.fromConfig(JSON.readTree("{\"url\":\"http://localhost:9090\"}"), "rtp"));
    }
}
