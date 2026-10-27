package org.flexlb.mock.runtime;

import static org.junit.jupiter.api.Assertions.*;
import java.net.*;
import java.net.http.*;
import java.nio.file.*;
import java.time.Instant;
import java.util.*;
import org.flexlb.mock.api.InterfaceRegistry;
import org.junit.jupiter.api.*;

class ApiServerTest {
    private ApiServer server;
    private static final HttpClient CLIENT = HttpClient.newHttpClient();

    @BeforeEach void start() throws Exception {
        server = new ApiServer(new InetSocketAddress("127.0.0.1", 0))
            .bind(TransportProbe.Api.class, new TransportProbe.Implementation()).start();
    }
    @AfterEach void close() { server.close(); }

    private HttpResponse<String> call(String route, Object args, String run, long timeout) throws Exception {
        var context = new HashMap<String, Object>();
        context.put("run", run); context.put("query_time", Instant.now().toString());
        context.put("observation", null); context.put("timeout_ms", timeout);
        var body = JsonWire.JSON.writeValueAsString(Map.of("interface", route, "args", args, "context", context));
        return CLIENT.send(HttpRequest.newBuilder(URI.create(server.url() + "/invoke"))
            .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
    }

    @Test void typedRecordsAndContext() throws Exception {
        var response = call("probe/action", Map.of(), "background", 1000);
        assertEquals(200, response.statusCode());
        assertEquals("master-0", JsonWire.JSON.readTree(response.body()).at("/result/affected/instances/0/id").asText());
        assertThrows(IllegalStateException.class, InvocationContext::current);
    }

    @Test void nativeOnlyNodeHealthAdvertisesNoImplementedBusinessInterfaces() throws Exception {
        try (var nativeHost = new ApiServer(new InetSocketAddress("127.0.0.1", 0)).start()) {
            var response = CLIENT.send(HttpRequest.newBuilder(URI.create(nativeHost.url() + "/health")).GET().build(), HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode());
            assertEquals(0, JsonWire.JSON.readTree(response.body()).get("interfaces").size());
        }
    }

    @Test void rejectsInvalidRequestsBeforeImplementation() throws Exception {
        assertEquals(400, call("probe/action", Map.of("delay_ms", -1), "foreground", 1000).statusCode());
        assertEquals(400, call("probe/action", Map.of("delay_ms", "1"), "foreground", 1000).statusCode());
        assertEquals(400, call("probe/action", Map.of("extra", 1), "foreground", 1000).statusCode());
        assertEquals(400, call("probe/metric", Map.of(), "background", 1000).statusCode());
        assertEquals(400, call("probe/missing", Map.of(), "foreground", 1000).statusCode());
    }

    @Test void validatesAndDecodesSpecialTypes() throws Exception {
        for (Object output : List.of(128, Map.of("distribution", "normal", "mean", 128, "stddev", 32, "min", 0, "max", 512, "seed", 42),
                Map.of("distribution", "choice", "values", List.of(1, 2), "weights", List.of(0, 1), "seed", 42)))
            assertEquals(200, call("probe/prepare", Map.of("output_tokens", output), "foreground", 1000).statusCode());
        assertEquals(400, call("probe/prepare", Map.of("output_tokens", Map.of("distribution", "uniform", "min", 5, "max", 1, "seed", 1)), "foreground", 1000).statusCode());
        assertEquals(200, call("probe/echo", Map.of("range", List.of("-60s", "0s")), "foreground", 1000).statusCode());
        assertEquals(400, call("probe/echo", Map.of("range", "all"), "foreground", 1000).statusCode());
    }

    @Test void timesOutWithoutRetry() throws Exception {
        assertEquals(504, call("probe/action", Map.of("delay_ms", 1000), "foreground", 20).statusCode());
        assertEquals(200, call("probe/metric", Map.of(), "foreground", 1000).statusCode());
    }

    @Test void rejectsInvalidComponentResult() throws Exception {
        assertEquals(500, call("probe/metric", Map.of("invalid_result", true), "foreground", 1000).statusCode());
    }

    @Test void exportsProbeCatalogAndClasspathForCrossProcessTests() throws Exception {
        Path target = Path.of("target"); Files.createDirectories(target);
        var catalog = new TreeMap<>(InterfaceRegistry.standard());
        catalog.putAll(InterfaceRegistry.describe(TransportProbe.Api.class));
        Files.write(target.resolve("probe-catalog.json"), JsonWire.JSON.writeValueAsBytes(Map.of("interfaces", catalog,
                "configuration", InterfaceRegistry.configuration())));
        Files.writeString(target.resolve("probe-classpath.txt"), System.getProperty("surefire.test.class.path", System.getProperty("java.class.path")));
    }
}
