package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for PRIORITY_MIX (priority:percent list, e.g. "70:10,60:15,50:50,40:15,30:10"):
 * spec parsing, weight-proportional assignment, and the legacy default — an
 * unset spec keeps priority 0 on every request (proto3 default, never
 * serialized) and the summary carries no priority_stats block.
 */
class PriorityMixTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    // ------------------------------------------------------------------
    // Parsing.
    // ------------------------------------------------------------------

    @Test
    void emptySpecParsesToNullLegacy() {
        assertNull(JavaLoadClient.PriorityMix.parse(null));
        assertNull(JavaLoadClient.PriorityMix.parse(""));
        assertNull(JavaLoadClient.PriorityMix.parse("   "));
    }

    @Test
    void parsesSpecWithWhitespace() {
        JavaLoadClient.PriorityMix mix =
                JavaLoadClient.PriorityMix.parse(" 70 : 10 , 30 : 90 ");
        assertEquals(100, mix.totalWeight);
        assertEquals(70, mix.priorityFor(0));
        assertEquals(70, mix.priorityFor(9));
        assertEquals(30, mix.priorityFor(10));
        assertEquals(30, mix.priorityFor(99));
    }

    @Test
    void rejectsMalformedSpecs() {
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("70"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("70:10:5"));
        // NumberFormatException is an IllegalArgumentException.
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("a:b"));
        // priority and percent must both be positive; 0 priority is reserved
        // for the legacy path and must not appear in a mix.
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("0:50,30:50"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("70:0"));
        assertThrows(IllegalArgumentException.class,
                () -> JavaLoadClient.PriorityMix.parse("-5:10"));
    }

    // ------------------------------------------------------------------
    // Weight-proportional assignment.
    // ------------------------------------------------------------------

    @Test
    void priorityForMapsRollsOntoCumulativeWeightBands() {
        JavaLoadClient.PriorityMix mix =
                JavaLoadClient.PriorityMix.parse("70:10,60:15,50:50,40:15,30:10");
        assertEquals(100, mix.totalWeight);
        // Exhaustive check: every roll lands in its band, so the assignment
        // matches the configured percentages exactly.
        Map<Integer, Integer> counts = new HashMap<>();
        for (int roll = 0; roll < mix.totalWeight; roll++) {
            counts.merge(mix.priorityFor(roll), 1, Integer::sum);
        }
        assertEquals(Map.of(70, 10, 60, 15, 50, 50, 40, 15, 30, 10), counts);
    }

    @Test
    void sampleFollowsConfiguredProportions() {
        JavaLoadClient.PriorityMix mix =
                JavaLoadClient.PriorityMix.parse("70:10,50:80,30:10");
        Random random = new Random(42);
        int n = 100_000;
        Map<Integer, Integer> counts = new HashMap<>();
        for (int i = 0; i < n; i++) {
            counts.merge(mix.sample(random), 1, Integer::sum);
        }
        assertEquals(3, counts.size(), counts.toString());
        // 3-sigma tolerance for a fair sampler at these proportions.
        assertTrue(Math.abs(counts.get(70) - n * 0.10) < n * 0.01, counts.toString());
        assertTrue(Math.abs(counts.get(50) - n * 0.80) < n * 0.01, counts.toString());
        assertTrue(Math.abs(counts.get(30) - n * 0.10) < n * 0.01, counts.toString());
    }

    // ------------------------------------------------------------------
    // End-to-end (dry-run): per_request priority + summary priority_stats.
    // ------------------------------------------------------------------

    private Path writeTrace(int n) throws IOException {
        Path trace = tempDir.resolve("trace.jsonl");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < n; i++) {
            sb.append("{\"request_id\":\"rid-").append(i)
                    .append("\",\"il\":32,\"ol\":4,\"ts\":").append(i * 10L)
                    .append("}\n");
        }
        Files.writeString(trace, sb.toString());
        return trace;
    }

    private JavaLoadClient.Config config(String traceFile, String outDir, String priorityMix) {
        return new JavaLoadClient.Config(
                traceFile, "127.0.0.1:7001", "127.0.0.1:7003",
                0, 16, 1000.0, 1, outDir, 1, 0, 0,
                120_000L, 500.0, "skip", false, false, 1, 1, 0L, 120, true,
                "engine_service", "", false,
                false, 10, 1000, 0, 0, "", false, "", true, "replay", 0.0, priorityMix);
    }

    private List<JsonNode> readPerRequest(Path outDir) throws IOException {
        List<JsonNode> rows = new ArrayList<>();
        for (String line : Files.readAllLines(outDir.resolve("per_request.jsonl"))) {
            if (!line.isBlank()) {
                rows.add(MAPPER.readTree(line));
            }
        }
        return rows;
    }

    @Test
    void mixRunTagsRequestsAndAggregatesPriorityStats() throws Exception {
        Path trace = writeTrace(200);
        Path outDir = tempDir.resolve("mix_out");
        JavaLoadClient client = new JavaLoadClient(
                config(trace.toString(), outDir.toString(), "70:50,30:50"));
        client.run();

        List<JsonNode> rows = readPerRequest(outDir);
        assertEquals(200, rows.size());
        Map<Integer, Integer> perRequestCounts = new HashMap<>();
        for (JsonNode row : rows) {
            int priority = row.get("priority").asInt();
            assertTrue(priority == 70 || priority == 30,
                    "priority outside mix: " + priority);
            perRequestCounts.merge(priority, 1, Integer::sum);
        }

        JsonNode summary = MAPPER.readTree(Files.readString(outDir.resolve("summary.json")));
        JsonNode stats = summary.get("priority_stats");
        assertTrue(stats != null && stats.isObject(), "summary lacks priority_stats");
        int total = 0;
        for (Map.Entry<Integer, Integer> entry : perRequestCounts.entrySet()) {
            JsonNode group = stats.get(String.valueOf(entry.getKey()));
            assertTrue(group != null, "priority_stats lacks group " + entry.getKey());
            assertEquals((int) entry.getValue(), group.get("total").asInt());
            // Dry-run requests all fail fast: success + fail must still
            // account for every request in the group.
            assertEquals(group.get("total").asInt(),
                    group.get("success").asInt() + group.get("fail").asInt());
            assertTrue(group.get("error_status_counts").isObject());
            total += group.get("total").asInt();
        }
        assertEquals(200, total);
    }

    @Test
    void legacyRunKeepsPriorityZeroAndOmitsPriorityStats() throws Exception {
        Path trace = writeTrace(20);
        Path outDir = tempDir.resolve("legacy_out");
        JavaLoadClient client = new JavaLoadClient(
                config(trace.toString(), outDir.toString(), ""));
        client.run();

        for (JsonNode row : readPerRequest(outDir)) {
            assertEquals(0, row.get("priority").asInt());
        }
        JsonNode summary = MAPPER.readTree(Files.readString(outDir.resolve("summary.json")));
        assertFalse(summary.has("priority_stats"),
                "legacy summary must not carry priority_stats");
    }
}
