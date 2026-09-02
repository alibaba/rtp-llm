package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the uniform (evenly-spaced arrival) send mode
 * (SEND_MODE=uniform, SEND_MODE_QPS=N).
 *
 * <p>Uniform mode only changes the arrival process: request bodies still cycle
 * through the trace shard with the loop-mode "_S{shard}_L{loop}" rid suffixes,
 * but ideal send times form a fixed-interval schedule t0 + i*interval where
 * interval = NUM_SHARDS / SEND_MODE_QPS. Verified end-to-end by driving
 * {@code run()} in dry-run mode (no gRPC channels; every request fails fast)
 * and inspecting the client_events.jsonl send_due_epoch_ms schedule.
 */
class UniformSendModeTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    private Path writeTrace(int n) throws IOException {
        Path trace = tempDir.resolve("trace.jsonl");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append("{\"request_id\":\"rid-").append(i)
                    .append("\",\"il\":32,\"ol\":4,\"ts\":").append(i * 1000L)
                    .append("}\n");
        }
        Files.writeString(trace, sb.toString());
        return trace;
    }

    private JavaLoadClient.Config config(String traceFile, String outDir,
                                         int durationS, int limit, int numShards, int shardIndex,
                                         boolean loop, String sendMode, double sendModeQps) {
        // Full constructor: priority 0 (unset) keeps the Auto-TPM QoS
        // passthrough path untouched by uniform-mode tests; forcePriority 0
        // likewise leaves per-record priority resolution alone.
        return new JavaLoadClient.Config(
                traceFile, "127.0.0.1:7001", "127.0.0.1:7003",
                durationS, 16, 1000.0, 1, outDir, numShards, shardIndex, limit,
                120_000L, 500.0, "skip", false, loop, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true, 0, 0, sendMode, sendModeQps,
                true);
    }

    private List<JsonNode> readPerRequest(Path outDir) throws IOException {
        List<JsonNode> rows = new ArrayList<>();
        for (String line : Files.readAllLines(outDir.resolve("client_events.jsonl"))) {
            if (!line.isBlank()) {
                rows.add(MAPPER.readTree(line));
            }
        }
        return rows;
    }

    // ------------------------------------------------------------------
    // Config validation: SEND_MODE / SEND_MODE_QPS.
    // ------------------------------------------------------------------

    @Test
    void rejectsUnknownSendMode() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> config("trace.jsonl", tempDir.resolve("out").toString(),
                        0, 0, 1, 0, false, "burst", 10.0));
        assertTrue(e.getMessage().contains("SEND_MODE"), e.getMessage());
    }

    @Test
    void uniformRequiresPositiveQps() {
        assertThrows(IllegalArgumentException.class,
                () -> config("trace.jsonl", tempDir.resolve("out").toString(),
                        0, 0, 1, 0, false, "uniform", 0.0));
        assertThrows(IllegalArgumentException.class,
                () -> config("trace.jsonl", tempDir.resolve("out").toString(),
                        0, 0, 1, 0, false, "uniform", -5.0));
    }

    @Test
    void replayIgnoresQpsAndStaysValid() {
        JavaLoadClient.Config replay = config("trace.jsonl", tempDir.resolve("out").toString(),
                0, 0, 1, 0, false, "replay", 0.0);
        assertFalse(replay.isUniform());
        JavaLoadClient.Config uniform = config("trace.jsonl", tempDir.resolve("out").toString(),
                0, 0, 1, 0, false, "uniform", 50.0);
        assertTrue(uniform.isUniform());
    }

    // ------------------------------------------------------------------
    // Uniform pacing: due times are evenly spaced, count = QPS x duration.
    // ------------------------------------------------------------------

    @Test
    void uniformSendsEvenlySpacedAtTargetQps() throws Exception {
        Path trace = writeTrace(5);
        Path outDir = tempDir.resolve("uniform_out");
        // 50 QPS for 2s -> 100 sends at exactly 20ms spacing (single shard).
        JavaLoadClient client = new JavaLoadClient(config(trace.toString(), outDir.toString(),
                2, 0, 1, 0, false, "uniform", 50.0));
        client.run();

        List<JsonNode> rows = readPerRequest(outDir);
        int expected = 50 * 2;
        assertTrue(rows.size() >= expected - 1 && rows.size() <= expected + 1,
                "expected " + expected + "±1 sends, got " + rows.size());

        List<Double> dues = new ArrayList<>();
        Set<String> rids = new HashSet<>();
        for (JsonNode row : rows) {
            dues.add(row.get("send_due_epoch_ms").asDouble());
            rids.add(row.get("rid").asText());
        }
        // Trace cycling keeps rids disjoint via the _S{shard}_L{loop} suffix.
        assertEquals(rows.size(), rids.size(), "duplicate rid in uniform loop mode");

        dues.sort(null);
        for (int i = 1; i < dues.size(); i++) {
            assertEquals(20.0, dues.get(i) - dues.get(i - 1), 1e-6,
                    "non-uniform interval at index " + i);
        }

        // The client records raw rows only (no summary.json): the achieved
        // send rate is recovered from send_due_epoch_ms itself.
        double spanS = (dues.get(dues.size() - 1) - dues.get(0)) / 1000.0;
        double achievedQps = (dues.size() - 1) / Math.max(spanS, 1e-9);
        assertEquals(50.0, achievedQps, 1.0, "achieved QPS far from target");
    }

    @Test
    void uniformSplitsQpsAcrossShards() throws Exception {
        Path trace = writeTrace(5);
        Path outDir = tempDir.resolve("shard_out");
        // 200 QPS over 4 shards -> 50 QPS per shard (20ms interval) for 1s.
        JavaLoadClient client = new JavaLoadClient(config(trace.toString(), outDir.toString(),
                1, 0, 4, 1, false, "uniform", 200.0));
        client.run();

        List<JsonNode> rows = readPerRequest(outDir);
        int expected = 50;
        assertTrue(rows.size() >= expected - 1 && rows.size() <= expected + 1,
                "expected " + expected + "±1 sends, got " + rows.size());

        List<Double> dues = new ArrayList<>();
        for (JsonNode row : rows) {
            dues.add(row.get("send_due_epoch_ms").asDouble());
        }
        dues.sort(null);
        for (int i = 1; i < dues.size(); i++) {
            assertEquals(20.0, dues.get(i) - dues.get(i - 1), 1e-6,
                    "non-uniform per-shard interval at index " + i);
        }

        // Per-shard rate is recovered from the raw dues (20ms interval =
        // 50 QPS on this shard; the 200 QPS total is the 4-shard sum).
        double spanS = (dues.get(dues.size() - 1) - dues.get(0)) / 1000.0;
        double perShardQps = (dues.size() - 1) / Math.max(spanS, 1e-9);
        assertEquals(50.0, perShardQps, 1.0, "per-shard QPS far from target");
    }

    @Test
    void uniformLimitCapsTotalSends() throws Exception {
        Path trace = writeTrace(3);
        Path outDir = tempDir.resolve("limit_out");
        // DURATION_S=0: LIMIT alone stops the run (loop-replay semantics).
        JavaLoadClient client = new JavaLoadClient(config(trace.toString(), outDir.toString(),
                0, 10, 1, 0, false, "uniform", 1000.0));
        client.run();

        assertEquals(10, readPerRequest(outDir).size());
    }

    // ------------------------------------------------------------------
    // Replay default path: behavior unchanged, raw output only.
    // ------------------------------------------------------------------

    @Test
    void replayPathUnchangedByDefault() throws Exception {
        Path trace = writeTrace(5);
        Path outDir = tempDir.resolve("replay_out");
        // Non-loop replay: all 5 trace records sent once, paced by trace ts.
        JavaLoadClient client = new JavaLoadClient(config(trace.toString(), outDir.toString(),
                0, 0, 1, 0, false, "replay", 0.0));
        client.run();

        List<JsonNode> rows = readPerRequest(outDir);
        assertEquals(5, rows.size());
        Set<String> rids = new HashSet<>();
        for (JsonNode row : rows) {
            rids.add(row.get("rid").asText());
        }
        // No loop suffixes on the plain replay path.
        for (String rid : rids) {
            assertFalse(rid.contains("_S"), "unexpected loop suffix: " + rid);
        }

        // The client writes no summary.json at all anymore: raw rows only.
        assertFalse(Files.exists(outDir.resolve("summary.json")),
                "client must not write summary.json");
    }
}
