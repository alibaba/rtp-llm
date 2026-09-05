package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Parity tests for JavaLoadClient against the Python reference implementation
 * (the legacy Python load client): shard ordering, gradient pacing,
 * length truncation, nearest-rank percentiles, fallback endpoint parsing and
 * pushgateway metric payload format.
 */
class JavaLoadClientParityTest {

    @TempDir
    Path tempDir;

    private static JavaLoadClient.TraceRecord rec(int idx, long tsMs, int il, int ol) {
        List<Integer> tokens = new ArrayList<>();
        for (int i = 0; i < il; i++) {
            tokens.add(i);
        }
        List<Long> keys = new ArrayList<>();
        for (long k = 0; k < il / 1024; k++) {
            keys.add(k + idx);
        }
        return new JavaLoadClient.TraceRecord(idx, "rid-" + idx, "trace-" + idx, tsMs,
                il, ol, keys, tokens);
    }

    private JavaLoadClient dryRunClient() {
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:7003",
                0, 16, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
                120_000L, 500.0, false, false, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true);
        return new JavaLoadClient(config);
    }

    // ------------------------------------------------------------------
    // Item 1: shard ordering — duration/limit filters BEFORE i % numShards.
    // ------------------------------------------------------------------

    @Test
    void limitAppliesToWholeTraceNotPerShard() {
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            records.add(rec(i, i * 1000L, 100, 10));
        }

        List<JavaLoadClient.TraceRecord> shard0 =
                JavaLoadClient.filterAndShard(records, 0, 5, 2, 0);
        List<JavaLoadClient.TraceRecord> shard1 =
                JavaLoadClient.filterAndShard(records, 0, 5, 2, 1);

        // Old (buggy) order would give LIMIT records per shard = 10 total.
        // Python order: limit first (5 records), then shard slice -> 5 total.
        assertEquals(5, shard0.size() + shard1.size());
        assertEquals(3, shard0.size());
        assertEquals(2, shard1.size());
        assertEquals(0, shard0.get(0).requestId);
        assertEquals(2, shard0.get(1).requestId);
        assertEquals(4, shard0.get(2).requestId);
        assertEquals(1, shard1.get(0).requestId);
        assertEquals(3, shard1.get(1).requestId);
    }

    @Test
    void durationFilterAppliesBeforeShard() {
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            records.add(rec(i, i * 1000L, 100, 10));
        }

        // duration 3s keeps ts <= firstTs + 3000 -> indices 0..3 (4 records),
        // then shard 0 of 2 -> indices 0, 2.
        List<JavaLoadClient.TraceRecord> shard0 =
                JavaLoadClient.filterAndShard(records, 3, 0, 2, 0);
        assertEquals(2, shard0.size());
        assertEquals(0, shard0.get(0).requestId);
        assertEquals(2, shard0.get(1).requestId);
    }

    @Test
    void noShardingWhenSingleShard() {
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            records.add(rec(i, i * 1000L, 100, 10));
        }
        List<JavaLoadClient.TraceRecord> out =
                JavaLoadClient.filterAndShard(records, 0, 4, 1, 0);
        assertEquals(4, out.size());
    }

    @Test
    void loopModeShardOnlySlicingCoversFullTraceDisjointly() {
        // Loop mode zeroes the duration/limit filters but must still slice the
        // trace across shards (Python parity): shards partition the full trace.
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 9; i++) {
            records.add(rec(i, i * 1000L, 100, 10));
        }
        List<JavaLoadClient.TraceRecord> shard0 =
                JavaLoadClient.filterAndShard(records, 0, 0, 4, 0);
        List<JavaLoadClient.TraceRecord> shard1 =
                JavaLoadClient.filterAndShard(records, 0, 0, 4, 1);
        List<JavaLoadClient.TraceRecord> shard2 =
                JavaLoadClient.filterAndShard(records, 0, 0, 4, 2);
        List<JavaLoadClient.TraceRecord> shard3 =
                JavaLoadClient.filterAndShard(records, 0, 0, 4, 3);

        assertEquals(9, shard0.size() + shard1.size() + shard2.size() + shard3.size());
        assertEquals(3, shard0.size());
        assertEquals(0, shard0.get(0).requestId);
        assertEquals(4, shard0.get(1).requestId);
        assertEquals(8, shard0.get(2).requestId);
        assertEquals(2, shard1.size());
        assertEquals(1, shard1.get(0).requestId);
        assertEquals(5, shard1.get(1).requestId);
    }

    // ------------------------------------------------------------------
    // Item 3: gradient pacing — linear ramp from start to max over duration.
    // ------------------------------------------------------------------

    @Test
    void gradientSpeedRampsLinearly() {
        assertEquals(10.0, JavaLoadClient.gradientSpeed(0.0, 60, 10, 1000), 1e-9);
        assertEquals(1000.0, JavaLoadClient.gradientSpeed(60.0, 60, 10, 1000), 1e-9);
        // halfway: 10 + (1000 - 10) * 0.5 = 505
        assertEquals(505.0, JavaLoadClient.gradientSpeed(30.0, 60, 10, 1000), 1e-9);
    }

    @Test
    void gradientSpeedClampsProgressAndStartSpeed() {
        // elapsed beyond duration clamps to max speed
        assertEquals(1000.0, JavaLoadClient.gradientSpeed(120.0, 60, 10, 1000), 1e-9);
        // negative elapsed treated as 0
        assertEquals(10.0, JavaLoadClient.gradientSpeed(-5.0, 60, 10, 1000), 1e-9);
        // start speed floored at 1 (Python: max(1, gradient_start_speed))
        assertEquals(1.0, JavaLoadClient.gradientSpeed(0.0, 60, 0, 1000), 1e-9);
    }

    // ------------------------------------------------------------------
    // Item 4: length truncation (Python L220-242 parity).
    // ------------------------------------------------------------------

    @Test
    void truncateRecordsCapsInputTokensAndOutputLen() {
        List<JavaLoadClient.TraceRecord> records = List.of(rec(0, 0, 5000, 300));
        List<JavaLoadClient.TraceRecord> out =
                JavaLoadClient.truncateRecords(records, 1000, 100);

        assertEquals(1, out.size());
        JavaLoadClient.TraceRecord r = out.get(0);
        assertEquals(1000, r.inputLen);
        assertEquals(1000, r.tokenIds.size());
        assertEquals(100, r.outputLen);
        // block_keys are NOT truncated (Python leaves them untouched)
        assertEquals(5000 / 1024, r.blockKeys.size());
        // identity fields preserved
        assertEquals("rid-0", r.sourceRid);
        assertEquals(0, (int) r.tokenIds.get(0));
        assertEquals(999, (int) r.tokenIds.get(999));
    }

    @Test
    void truncateRecordsLeavesShortRequestsUntouched() {
        List<JavaLoadClient.TraceRecord> records = List.of(rec(1, 0, 500, 50));
        List<JavaLoadClient.TraceRecord> out =
                JavaLoadClient.truncateRecords(records, 1000, 100);
        JavaLoadClient.TraceRecord r = out.get(0);
        assertEquals(500, r.inputLen);
        assertEquals(500, r.tokenIds.size());
        assertEquals(50, r.outputLen);
        // unmodified record passes through as the same instance
        assertTrue(r == records.get(0));
    }

    @Test
    void truncateRecordsNoopWhenBothCapsZero() {
        List<JavaLoadClient.TraceRecord> records = List.of(rec(2, 0, 2000, 200));
        List<JavaLoadClient.TraceRecord> out =
                JavaLoadClient.truncateRecords(records, 0, 0);
        assertEquals(2000, out.get(0).inputLen);
        assertEquals(200, out.get(0).outputLen);
    }

    // ------------------------------------------------------------------
    // Item 2: nearest-rank percentile (Python LoadClient._percentile).
    // ------------------------------------------------------------------

    @Test
    void nearestRankPercentileMatchesPython() {
        List<Double> values = List.of(10.0, 20.0, 30.0, 40.0);
        // Python: idx = int(len * p / 100); sorted[idx]
        assertEquals(30.0, JavaLoadClient.percentileNearestRank(values, 50), 1e-9);
        assertEquals(40.0, JavaLoadClient.percentileNearestRank(values, 99), 1e-9);
        assertEquals(10.0, JavaLoadClient.percentileNearestRank(values, 0), 1e-9);
        // nearest-rank differs from linear interpolation (which would give 25.0)
        assertEquals(30.0, JavaLoadClient.percentileNearestRank(values, 50), 1e-9);
    }

    @Test
    void nearestRankPercentileEdgeCases() {
        assertEquals(0.0, JavaLoadClient.percentileNearestRank(List.of(), 99), 1e-9);
        assertEquals(7.0, JavaLoadClient.percentileNearestRank(List.of(7.0), 50), 1e-9);
        // p99 on 100 values: idx = int(100 * 0.99) = 99 -> last element
        List<Double> hundred = new ArrayList<>();
        for (int i = 1; i <= 100; i++) {
            hundred.add((double) i);
        }
        assertEquals(100.0, JavaLoadClient.percentileNearestRank(hundred, 99), 1e-9);
        // idx = int(100 * 0.50) = 50 -> element 51 (nearest-rank, Python parity)
        assertEquals(51.0, JavaLoadClient.percentileNearestRank(hundred, 50), 1e-9);
        // unsorted input is sorted internally
        assertEquals(40.0, JavaLoadClient.percentileNearestRank(List.of(40.0, 10.0, 30.0, 20.0), 99), 1e-9);
    }

    // ------------------------------------------------------------------
    // Item 6: fallback endpoints.json parsing (Python _load_fallback_endpoints).
    // ------------------------------------------------------------------

    @Test
    void fallbackEndpointsFromDomainAddressEnv() throws Exception {
        Path endpoints = tempDir.resolve("endpoints.json");
        Files.writeString(endpoints, "{"
                + "\"prefill_domain\": \"mock.prefill.hosts.address\","
                + "\"decode_domain\": \"mock.decode.hosts.address\","
                + "\"env\": {"
                + "\"DOMAIN_ADDRESS:mock.prefill.hosts.address\": \"127.0.0.1:8001, 127.0.0.1:8002\","
                + "\"DOMAIN_ADDRESS:mock.decode.hosts.address\": \"127.0.0.1:9001\""
                + "}, \"engines\": []}");

        JavaLoadClient client = dryRunClient();
        client.loadFallbackEndpoints(endpoints.toString());

        // HTTP port + 1 = gRPC port (CommonConstants.GRPC_PORT_OFFSET)
        assertEquals(List.of("127.0.0.1:8002", "127.0.0.1:8003"), client.fallbackPrefillAddrs);
        assertEquals(List.of("127.0.0.1:9002"), client.fallbackDecodeAddrs);
    }

    @Test
    void fallbackEndpointsFromEnginesArray() throws Exception {
        Path endpoints = tempDir.resolve("endpoints.json");
        Files.writeString(endpoints, "{"
                + "\"prefill_domain\": \"p\", \"decode_domain\": \"d\", \"env\": {},"
                + "\"engines\": ["
                + "{\"role\": \"prefill\", \"grpc_addr\": \"127.0.0.1:6001\"},"
                + "{\"role\": \"decode\", \"grpc_addr\": \"127.0.0.1:6002\"},"
                + "{\"role\": \"prefill\", \"grpc_addr\": \"\"}"
                + "]}");

        JavaLoadClient client = dryRunClient();
        client.loadFallbackEndpoints(endpoints.toString());

        assertEquals(List.of("127.0.0.1:6001"), client.fallbackPrefillAddrs);
        assertEquals(List.of("127.0.0.1:6002"), client.fallbackDecodeAddrs);
    }

    // ------------------------------------------------------------------
    // Item 5: pushgateway payload format (Python _push_metrics parity).
    // ------------------------------------------------------------------

    @Test
    void pushMetricsBodyMatchesPythonMetricSet() {
        JavaLoadClient client = dryRunClient();
        client.sentTotal.set(10);
        client.actualSentCount.set(9);
        client.successCount.set(7);
        client.errorCount.set(1);
        client.inflightCount.set(2);

        JavaLoadClient.RequestResult ok = new JavaLoadClient.RequestResult();
        ok.status = "ok";
        ok.routePath = "master";
        ok.scheduleMs = 1.5;
        ok.totalMs = 20.0;
        ok.ttftMs = 10.0;
        client.completedResults.add(ok);

        JavaLoadClient.RequestResult fb = new JavaLoadClient.RequestResult();
        fb.status = "ok";
        fb.routePath = "fallback";
        fb.totalMs = 30.0;
        fb.ttftMs = 15.0;
        client.completedResults.add(fb);

        String body = client.buildPushMetricsBody();

        assertTrue(body.contains("flexlb_client_send_total{route_path=\"master\"} 10"), body);
        assertTrue(body.contains("flexlb_client_actual_send_total{route_path=\"master\"} 9"), body);
        assertTrue(body.contains("flexlb_client_completed_total{route_path=\"master\"} 2"), body);
        assertTrue(body.contains("flexlb_client_success_total{route_path=\"master\"} 7"), body);
        assertTrue(body.contains("flexlb_client_error_total{route_path=\"master\"} 1"), body);
        assertTrue(body.contains("flexlb_client_inflight_count{route_path=\"master\"} 2"), body);
        assertTrue(body.contains("flexlb_client_max_concurrency{route_path=\"master\"} 16"), body);
        // 2 / 16 = 0.1250
        assertTrue(body.contains("flexlb_client_semaphore_utilization{route_path=\"master\"} 0.1250"), body);
        // per-route_path latency groups with nearest-rank p50/p99
        assertTrue(body.contains("flexlb_client_total_ms_avg{route_path=\"master\"} 20.000"), body);
        assertTrue(body.contains("flexlb_client_total_ms_p50{route_path=\"master\"} 20.000"), body);
        assertTrue(body.contains("flexlb_client_total_ms_p99{route_path=\"master\"} 20.000"), body);
        assertTrue(body.contains("flexlb_client_total_ms_max{route_path=\"master\"} 20.000"), body);
        assertTrue(body.contains("flexlb_client_total_ms_count{route_path=\"master\"} 1"), body);
        assertTrue(body.contains("flexlb_client_ttft_ms_avg{route_path=\"fallback\"} 15.000"), body);
        // fallback result has schedule_ms=0 -> no schedule_ms metrics for fallback
        assertFalse(body.contains("flexlb_client_schedule_ms_count{route_path=\"fallback\"}"), body);
        assertTrue(body.endsWith("\n"));
    }

    @Test
    void pushMetricsBodyWithoutResultsStillPushesCounters() {
        JavaLoadClient client = dryRunClient();
        String body = client.buildPushMetricsBody();
        assertTrue(body.contains("flexlb_client_send_total{route_path=\"master\"} 0"));
        assertTrue(body.contains("flexlb_client_max_concurrency{route_path=\"master\"} 16"));
        assertFalse(body.contains("flexlb_client_total_ms_avg"));
    }
}
