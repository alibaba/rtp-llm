package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Emission-surface guard for the two telemetry outputs downstream tooling
 * parses by KEY NAME (P2-4): the {@code java_mock_stats} stdout line and the
 * {@code /snapshot} HTTP response. Only key presence is asserted — values are
 * runtime-dependent (heap, timestamps) and belong to behavior tests, not this
 * schema pin. A silent key rename would otherwise only surface as NaN columns
 * in aggregate_canvas_run.py / run analysis notebooks long after the run.
 */
class TelemetryEmissionSurfaceTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @Test
    void statsLineCarriesAllParsedKeys() {
        String line = JavaMockEngineCluster.buildStatsLine(
                List.of(), new JavaMockEngineCluster.ClusterStats());

        assertTrue(line.startsWith("java_mock_stats "), "line prefix is the parser anchor");
        // Keys consumed by run tooling (grep/awk on key=value pairs). Assert
        // presence of "key=" only — never the value.
        for (String key : new String[] {
                "ts_epoch_ms", "enqueue_rpcs", "enqueued_requests", "status_rpcs", "cache_rpcs",
                "prefill_batches", "avg_batch_size", "max_batch_size", "avg_batch_ms", "max_batch_ms",
                "prefill_exec_p50", "prefill_exec_p95",
                "prefill_waiting", "prefill_running", "prefill_running_reqs", "max_prefill_waiting",
                "decode_waiting", "decode_running", "decode_run_min", "decode_run_max",
                "max_decode_waiting", "decode_admitted", "decode_done", "decode_exec_p50",
                "decode_exec_p95",
                "decode_exec_max", "heap_used_mb", "heap_max_mb",
                "generate_stream_rpcs", "fetch_response_rpcs", "cancel_rpcs",
                "cancel_census_tracked", "cancel_census_finished", "cancel_census_unknown",
                "cancel_census_tombstone"}) {
            assertTrue(line.contains(" " + key + "=") || line.contains("java_mock_stats " + key + "="),
                    "stats line must carry key '" + key + "': " + line);
        }
    }

    @Test
    void snapshotEndpointCarriesTimestampAndEngines() throws Exception {
        MockControlServer controlServer = new MockControlServer(
                new ConcurrentHashMap<>(), new ConcurrentHashMap<>(), null, null, "127.0.0.1", 0);
        controlServer.start();
        try {
            HttpResponse<String> response = HttpClient.newHttpClient().send(
                    HttpRequest.newBuilder()
                            .uri(URI.create("http://127.0.0.1:" + controlServer.getPort() + "/snapshot"))
                            .GET()
                            .build(),
                    HttpResponse.BodyHandlers.ofString());
            assertEquals(200, response.statusCode());
            JsonNode json = MAPPER.readTree(response.body());
            assertTrue(json.has("ts_epoch_ms"),
                    "/snapshot must carry the ts_epoch_ms alignment timestamp");
            assertTrue(json.has("engines"), "/snapshot must carry the engines list");
        } finally {
            controlServer.stop();
        }
    }
}
