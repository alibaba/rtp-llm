package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Tests for the uniform-mode traffic ramp-up (RAMP_UP_SECONDS).
 *
 * <p>Two independent knobs are deliberately kept separate:
 * <ul>
 *   <li>FLEXLB_WARMUP_SECONDS — orchestrator-level prepare sleep with zero
 *       traffic before load starts (default on, 10s; lives in
 *       run_online_eval.sh, not in the client);</li>
 *   <li>RAMP_UP_SECONDS — arrival-process shaping inside uniform mode: the
 *       per-shard QPS climbs linearly from 0 to the target over N seconds,
 *       then holds constant (default 0 = off, legacy fixed interval).</li>
 * </ul>
 *
 * <p>The ideal ramped schedule inverts the triangle-integral arrival count
 * {@code N(t) = perShardQps * t^2 / (2 * ramp)} (see
 * {@code JavaLoadClient.uniformDueSeconds}), so pacing quality is conserved:
 * the ramp neither drops nor adds sends.
 */
class UniformRampUpTest {

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
                                         boolean loop, String sendMode, double sendModeQps,
                                         double rampUpSeconds) {
        return new JavaLoadClient.Config(
                traceFile, "127.0.0.1:7001", "127.0.0.1:7003",
                durationS, 16, 1000.0, 1, outDir, numShards, shardIndex, limit,
                120_000L, 500.0, false, loop, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true, 0, 0, sendMode, sendModeQps,
                rampUpSeconds, true);
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

    /** Due times in seconds relative to the first send (sorted). */
    private List<Double> readRelativeDues(Path outDir) throws IOException {
        List<Double> dues = new ArrayList<>();
        for (JsonNode row : readPerRequest(outDir)) {
            dues.add(row.get("send_due_epoch_ms").asDouble());
        }
        dues.sort(null);
        double t0 = dues.get(0);
        List<Double> rel = new ArrayList<>();
        for (double d : dues) {
            rel.add((d - t0) / 1000.0);
        }
        return rel;
    }

    // ------------------------------------------------------------------
    // Config validation.
    // ------------------------------------------------------------------

    @Test
    void negativeRampUpSecondsRejected() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> config("trace.jsonl", tempDir.resolve("out").toString(),
                        0, 0, 1, 0, false, "uniform", 50.0, -1.0));
        assertTrue(e.getMessage().contains("RAMP_UP_SECONDS"), e.getMessage());
    }

    // ------------------------------------------------------------------
    // Pure schedule math: JavaLoadClient.uniformDueSeconds.
    // ------------------------------------------------------------------

    @Test
    void scheduleDegeneratesToFixedIntervalWhenRampDisabled() {
        double qps = 100.0;
        for (int i = 0; i <= 500; i++) {
            assertEquals(i / qps, JavaLoadClient.uniformDueSeconds(i, qps, 0.0), 1e-12,
                    "fixed-interval mismatch at index " + i);
        }
    }

    @Test
    void scheduleConservesTriangleIntegralAtRampEnd() {
        // perShardQps=100, ramp=10s: ramped sends = triangle = 100*10/2 = 500,
        // and request #500 is due exactly at the 10s boundary (the two
        // schedule pieces meet continuously).
        double qps = 100.0, ramp = 10.0;
        assertEquals(10.0, JavaLoadClient.uniformDueSeconds(500, qps, ramp), 1e-9);
        assertTrue(JavaLoadClient.uniformDueSeconds(499, qps, ramp) < 10.0,
                "last ramp-phase request must still be due before the boundary");
        // First steady-phase request resumes at exactly the fixed interval.
        assertEquals(10.0 + 1.0 / qps,
                JavaLoadClient.uniformDueSeconds(501, qps, ramp), 1e-9);
    }

    @Test
    void instantaneousRateMatchesLinearCurve() {
        // For N(t) = a*t^2 the secant slope between consecutive due times
        // equals the tangent slope at the midpoint, so the discrete rate
        // 1/(t_{i+1} - t_i) matches the linear curve
        // q(t) = perShardQps * t / ramp for every ramp-phase pair.
        double qps = 100.0, ramp = 10.0;
        for (int i = 0; i < 499; i++) {
            double t0 = JavaLoadClient.uniformDueSeconds(i, qps, ramp);
            double t1 = JavaLoadClient.uniformDueSeconds(i + 1, qps, ramp);
            double discreteRate = 1.0 / (t1 - t0);
            double expected = qps * ((t0 + t1) / 2.0) / ramp;
            assertEquals(expected, discreteRate, 1e-9,
                    "instantaneous rate mismatch at ramp index " + i);
        }
        // Steady phase: the interval is exactly 1/qps again.
        double t500 = JavaLoadClient.uniformDueSeconds(500, qps, ramp);
        double t501 = JavaLoadClient.uniformDueSeconds(501, qps, ramp);
        assertEquals(1.0 / qps, t501 - t500, 1e-9);
    }

    // ------------------------------------------------------------------
    // End-to-end: run() in dry-run mode, inspect client_events.jsonl.
    // ------------------------------------------------------------------

    @Test
    void rampUpEndToEndConservesSendCountAndShape() throws Exception {
        Path trace = writeTrace(5);
        Path outDir = tempDir.resolve("ramp_out");
        // 100 QPS single shard, 2s linear ramp, 3s total:
        //   ramp window sends ≈ 100*2/2 = 100 (triangle integral),
        //   steady window adds ≈ 100, total ≈ 200.
        JavaLoadClient client = new JavaLoadClient(config(trace.toString(), outDir.toString(),
                3, 0, 1, 0, false, "uniform", 100.0, 2.0));
        client.run();

        List<Double> dues = readRelativeDues(outDir);
        assertTrue(dues.size() >= 195 && dues.size() <= 205,
                "expected ~200 sends (100 ramp + 100 steady), got " + dues.size());

        long rampSends = dues.stream().filter(d -> d <= 2.0).count();
        assertTrue(rampSends >= 98 && rampSends <= 102,
                "expected ~100 sends within the 2s ramp window, got " + rampSends);

        // Ramp phase: intervals shrink over time (rate climbs linearly).
        double early = dues.get(9) - dues.get(8);
        double late = dues.get(99) - dues.get(98);
        assertTrue(early > late,
                "ramp interval must shrink: early=" + early + "s late=" + late + "s");

        // Steady phase: last ~50 intervals are the fixed 10ms again.
        for (int i = Math.max(1, dues.size() - 50); i < dues.size(); i++) {
            assertEquals(0.01, dues.get(i) - dues.get(i - 1), 1e-6,
                    "non-fixed interval in steady phase at index " + i);
        }

        // The ramp shape itself is fully recovered from the raw dues; the
        // client writes no summary.json anymore.
        assertFalse(Files.exists(outDir.resolve("summary.json")),
                "client must not write summary.json");
    }

    @Test
    void zeroRampUpKeepsLegacyUniformSchedule() throws Exception {
        Path trace = writeTrace(5);
        Path outDir = tempDir.resolve("noramp_out");
        // RAMP_UP_SECONDS=0: fixed 20ms interval, byte-identical to the
        // pre-ramp uniform behavior; 50 QPS for 2s -> 100 sends.
        JavaLoadClient client = new JavaLoadClient(config(trace.toString(), outDir.toString(),
                2, 0, 1, 0, false, "uniform", 50.0, 0.0));
        client.run();

        List<Double> dues = readRelativeDues(outDir);
        int expected = 50 * 2;
        assertTrue(dues.size() >= expected - 1 && dues.size() <= expected + 1,
                "expected " + expected + "±1 sends, got " + dues.size());
        for (int i = 1; i < dues.size(); i++) {
            assertEquals(0.02, dues.get(i) - dues.get(i - 1), 1e-6,
                    "non-uniform interval at index " + i);
        }
    }
}
