package org.flexlb.service.lease;

import com.sun.net.httpserver.HttpServer;
import org.flexlb.config.PrefillLeaseConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.net.InetSocketAddress;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillTtftMetricReaderTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void readsExistingSumCountAndRejectsStaleSumThroughHttp(boolean staleSum) throws Exception {
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        List<String> queries = new ArrayList<>();
        server.createContext("/api/v1/query", exchange -> {
            String query = URLDecoder.decode(exchange.getRequestURI().getRawQuery(), StandardCharsets.UTF_8);
            queries.add(query);
            String number = query.contains("timestamp(")
                    ? staleSum && query.contains("_sum") ? "1" : "99"
                    : query.contains("_sum") ? "2" : "10";
            byte[] body = ("{\"status\":\"success\",\"data\":{\"resultType\":\"vector\",\"result\":["
                    + "{\"metric\":{\"lease_node\":\"pod-a\"},\"value\":[100,\"" + number + "\"]}]}}")
                    .getBytes(StandardCharsets.UTF_8);
            exchange.sendResponseHeaders(200, body.length);
            try (var output = exchange.getResponseBody()) {
                output.write(body);
            }
        });
        server.start();
        try {
            PrefillLeaseConfig config = new PrefillLeaseConfig();
            config.setMetricStoreUrl("http://127.0.0.1:" + server.getAddress().getPort());
            config.setWorkerMetricNodes(Map.of("10.0.0.1:8000@0", "pod-a"));
            if (staleSum) {
                assertThrows(IllegalArgumentException.class,
                        () -> new PrefillTtftMetricReader().read(config, List.of("10.0.0.1:8000@0"), 100_000));
            } else {
                var result = new PrefillTtftMetricReader().read(config, List.of("10.0.0.1:8000@0"), 100_000);
                assertEquals(200.0, result.meanMs(), 0.0001);
                assertEquals(10.0, result.samples());
                assertEquals(99_000, result.oldestSampleMs());
            }
            assertEquals(4, queries.size());
            assertTrue(queries.stream().allMatch(query -> query.contains("pod-a") && query.contains("time=100.0")));
        } finally {
            server.stop(0);
        }
    }

    @Test
    void weightsByRequestsRatherThanAveragingNodeMeans() {
        var result = PrefillTtftMetricReader.combine(Set.of("a", "b"), Map.of("a", 1.0, "b", 9.0),
                Map.of("a", 10.0, "b", 30.0), Map.of("a", 99.0, "b", 98.0), 100_000, 10);
        assertEquals(250.0, result.meanMs(), 0.0001);
        assertEquals(40.0, result.samples());
        assertEquals(98_000, result.oldestSampleMs());
    }

    @Test
    void zeroNewSamplesDoNotDiluteMean() {
        var result = PrefillTtftMetricReader.combine(Set.of("a", "idle"), Map.of("a", 2.0, "idle", 0.0),
                Map.of("a", 10.0, "idle", 0.0), Map.of("a", 99.0, "idle", 99.0), 100_000, 10);
        assertEquals(200.0, result.meanMs(), 0.0001);
    }

    @Test
    void rejectsMissingNodeAndStaleOrInvalidSamples() {
        assertThrows(IllegalArgumentException.class, () -> PrefillTtftMetricReader.combine(Set.of("a", "missing"),
                Map.of("a", 1.0), Map.of("a", 1.0), Map.of("a", 99.0), 100_000, 10));
        for (double stamp : new double[]{80.0, 102.0, Double.NaN}) {
            assertThrows(IllegalArgumentException.class, () -> PrefillTtftMetricReader.combine(Set.of("a"),
                    Map.of("a", 1.0), Map.of("a", 1.0), Map.of("a", stamp), 100_000, 10));
        }
        assertThrows(IllegalArgumentException.class, () -> PrefillTtftMetricReader.combine(Set.of("a"),
                Map.of("a", 1.0), Map.of("a", 0.0), Map.of("a", 99.0), 100_000, 10));
        assertThrows(IllegalArgumentException.class, () -> PrefillTtftMetricReader.combine(Set.of("a"),
                Map.of("a", Double.NaN), Map.of("a", 1.0), Map.of("a", 99.0), 100_000, 10));
    }

    @Test
    void selectorEscapesNodeRegexAndScopeLabelValues() {
        PrefillLeaseConfig config = new PrefillLeaseConfig();
        config.setMetricLabels(Map.of("deployment", "test\"name"));
        String selector = PrefillTtftMetricReader.selector(config, Set.of("pod.1+", "pod(2)"));
        assertTrue(selector.contains("pod\\\\.1\\\\+"));
        assertTrue(selector.contains("pod\\\\(2\\\\)"));
        assertTrue(selector.contains("deployment=\"test\\\"name\""));
    }
}
