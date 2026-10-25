package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** Real execution TPS through the HTTP API, separate from business completions. */
class TpsMetricsAccountingTest {
    @TempDir Path tempDir;

    private static double value(String body, String name, String labels) {
        String prefix = name + "{" + labels + "} ";
        return body.lines().filter(line -> line.startsWith(prefix))
                .mapToDouble(line -> Double.parseDouble(line.substring(prefix.length()))).findFirst().orElseThrow();
    }

    private static void awaitTps(JavaMockEngineCluster.FastRpcService service, long tokens) throws Exception {
        long deadline = System.nanoTime() + 5_000_000_000L;
        while (service.prefillTpsSnapshot().inputTokens() < tokens && System.nanoTime() < deadline) Thread.sleep(5);
        assertEquals(tokens, service.prefillTpsSnapshot().inputTokens());
    }

    @Test
    void coldAndCachedBatchesUseTheirOwnExecutionDenominators() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(tempDir, "10"), 62900, 1, 1)) {
            var p = cluster.prefill(0);
            var keys = List.of(11L, 12L, 13L);
            enqueueAndFetch(p, batch(1000, slot(0, inputWithBlockKeys(100, 3072, keys))));
            cluster.awaitCompleted(1, 5000);
            cluster.awaitAllInflightZero(2000);
            enqueueAndFetch(p, batch(1001, slot(0, inputWithBlockKeys(101, 3072, keys))));
            cluster.awaitCompleted(2, 5000);
            awaitTps(p, 6144);
            var totals = p.prefillTpsSnapshot();
            assertEquals(3072, totals.computeTokens());
            assertTrue(totals.inputUs() > totals.computeUs());
            String body = httpGet(cluster.controlPort(), "/metrics?per_engine=true");
            String labels = "engine_name=\"" + p.getEngineName() + "\",role=\"prefill\",grpc_port=\"" + p.getGrpcPort()
                    + "\",engine_ip=\"127.0.0.1\",engine_incarnation=\"" + p.getSnapshot().get("engine_incarnation") + "\"";
            assertEquals(3072 * 1e6 / totals.computeUs(), value(body, "rtp_llm_context_tps", labels), 0.000001);
            assertEquals(6144 * 1e6 / totals.inputUs(), value(body, "rtp_llm_context_tps_with_cache", labels), 0.000001);
            double wallUs = value(body, "rtp_llm_wall_tps_report_interval_us", labels);
            assertEquals(6144 * 1e6 / wallUs, value(body, "rtp_llm_context_wall_tps_with_cache", labels), 0.000001);
            assertEquals(3072L, p.getSnapshot().get("hit_tokens_total"));
            assertEquals(0.0, value(httpGet(cluster.controlPort(), "/metrics"),
                    "rtp_llm_context_tps", "role=\"prefill\""));
            assertFalse(body.contains("rtp_llm_context_tps{engine_name=\"decode"));
        }
    }

    @Test
    void aggregatedModeSumsPerEngineRatesNotRawTokenCounts() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(tempDir, "20"), 62920, 2, 1)) {
            double expected = 0;
            for (int i = 0; i < 2; i++) {
                var p = cluster.prefill(i);
                int tokens = (i + 1) * 1000;
                enqueueAndFetch(p, batch(2000 + i, slot(0, inputWithBlockKeys(200 + i, tokens, List.of()))));
                awaitTps(p, tokens);
                expected += tokens * 1e6 / p.prefillTpsSnapshot().computeUs();
            }
            String body = httpGet(cluster.controlPort(), "/metrics");
            assertEquals(expected, value(body, "rtp_llm_context_tps", "role=\"prefill\""), 0.000001);
        }
    }

    @Test
    void cancellationAfterStartDoesNotEraseExecutedTokens() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(tempDir, "300"), 62940, 1, 1)) {
            var p = cluster.prefill(0);
            enqueueAndFetch(p, batch(3000, slot(0,
                    inputWithBlockKeys(300, 512, List.of()), inputWithBlockKeys(301, 512, List.of()))));
            assertEquals(1, p.prefillTpsSnapshot().active());
            p.cancel(301L);
            awaitTps(p, 1024);
            assertEquals(1024, p.prefillTpsSnapshot().computeTokens());
            assertEquals(1024, p.cacheHitSnapshot().inputTokens(),
                    "cache reuse observations survive cancellation after execution starts");
            assertEquals(512L, p.getSnapshot().get("context_tokens_total"),
                    "business completion counter remains separate from executed work");
        }
    }

    @Test
    void whaleAndHttpUseTheSameAtomicLedger() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(tempDir, "10"), 62960, 1, 1)) {
            var p = cluster.prefill(0);
            enqueueAndFetch(p, batch(4000, slot(0, inputWithBlockKeys(400, 1000, List.of()))));
            awaitTps(p, 1000);
            Map<String, Double> reported = new HashMap<>();
            var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                    getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                    (proxy, method, args) -> {
                        if (method.getName().equals("report") && args.length == 3)
                            reported.put((String) args[0], ((Number) args[2]).doubleValue());
                        return null;
                    });
            var monitor = new WhaleMockMonitor(sink);
            monitor.sample(p); // First observation must include the already completed batch.
            String body = httpGet(cluster.controlPort(), "/metrics");
            assertEquals(reported.get("rtp_llm_context_tps"),
                    value(body, "rtp_llm_context_tps", "role=\"prefill\""), 0.000001);
            assertEquals(reported.get("rtp_llm_context_tps_with_cache"),
                    value(body, "rtp_llm_context_tps_with_cache", "role=\"prefill\""), 0.000001);
        }
    }
    @Test
    void decodeHttpRateUsesExecutedWorkAndElapsedTime() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(tempDir, "10"), 63040, 1, 1)) {
            var d = cluster.decode(0);
            enqueueAndFetch(cluster.prefill(0), batch(5000, slot(0,
                    inputWithDecode(500, 100, d.getGrpcPort(), 12))));
            cluster.awaitCompleted(1, 5000);
            cluster.awaitAllInflightZero(2000);
            var work = d.decodeTpsSnapshot();
            assertTrue(work.tokens() > 0);
            long before = System.nanoTime();
            String body = httpGet(cluster.controlPort(), "/metrics");
            long after = System.nanoTime();
            double rate = value(body, "mock_decode_wall_tps", "role=\"decode\"");
            assertTrue(rate >= CounterRateMetrics.rate(work.tokens(), after - work.startedNanos()) - 0.000001);
            assertTrue(rate <= CounterRateMetrics.rate(work.tokens(), before - work.startedNanos()) + 0.000001);
            assertEquals(work.tokens(), value(body, "rtp_llm_generate_tps", "role=\"decode\""));
            assertEquals(work.tokens(), value(body, "mock_decode_step_tokens_total", "role=\"decode\""));
            // A new HTTP scrape sees no new execution; the Whale reader still sees it.
            assertEquals(0.0, value(httpGet(cluster.controlPort(), "/metrics"),
                    "mock_decode_wall_tps", "role=\"decode\""));
            Map<String, Double> reports = new HashMap<>();
            var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                    getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                    (proxy, method, args) -> {
                        if (method.getName().equals("report") && args.length == 3)
                            reports.put((String) args[0], ((Number) args[2]).doubleValue());
                        return null;
                    });
            var monitor = new WhaleMockMonitor(sink);
            before = System.nanoTime();
            monitor.sample(d);
            after = System.nanoTime();
            assertEquals((double) work.tokens(), reports.get("rtp_llm_generate_tps"));
            rate = reports.get("mock_decode_wall_tps");
            assertTrue(rate >= CounterRateMetrics.rate(work.tokens(), after - work.startedNanos()));
            assertTrue(rate <= CounterRateMetrics.rate(work.tokens(), before - work.startedNanos()));
        }
    }

}
