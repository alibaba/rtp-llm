package org.flexlb.mockengine;

import java.util.ArrayList;
import java.util.Map;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MockMetricOwnershipTest {
    @org.junit.jupiter.api.io.TempDir java.nio.file.Path directory;

    @Test void decodeReuseMeasuresItsOwnLeaseOnceRatherThanPrefillShapeHits() throws Exception {
        var perf = directory.resolve("performance.json");
        var master = directory.resolve("master.json");
        java.nio.file.Files.writeString(perf, "{\"block_size\":64,\"prefill\":{\"formula\":\"1\"},\"decode\":{}}");
        java.nio.file.Files.writeString(master, "{}");
        var model = MockPerformanceModel.load(perf.toString(), master.toString());
        try (var cluster = MockEngineTestCluster.create(model, 63180, 0, 1)) {
            var d = cluster.decode(0);
            var field = d.getClass().getDeclaredField("cache");
            field.setAccessible(true);
            var cache = (MockLruBlockCache) field.get(d);
            cache.admit(java.util.List.of(7L));
            var events = new java.util.concurrent.CopyOnWriteArrayList<Map<String, Number>>();
            var reporter = d.getClass().getDeclaredField("eventMetricReporter");
            reporter.setAccessible(true);
            reporter.set(d, (java.util.function.Consumer<Map<String, Number>>) events::add);
            var input = org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder().setRequestId(42).build();
            // P claims two hit blocks, but D has only one of those blocks cached.
            var shape = new MockPerformanceModel.RequestShape(input, 128, 1,
                    java.util.List.of(7L, 8L), 128, 2, false);
            assertTrue(d.reserveDecodeLease(42, shape));
            assertTrue(d.reserveDecodeLease(42, shape));
            var reuse = events.stream().filter(m -> m.containsKey("rtp_llm_kv_cache_reuse_length")).toList();
            assertEquals(1, reuse.size(), "adopting a reservation must not double-report reuse");
            assertEquals(64L, reuse.get(0).get("rtp_llm_kv_cache_reuse_length"));
            assertEquals(64L, reuse.get(0).get("rtp_llm_stream_cache_device_reuse_length"));
            assertEquals(0L, reuse.get(0).get("rtp_llm_stream_cache_memory_reuse_length"));
            assertEquals(128, d.cacheHitSnapshot().inputTokens());
            assertEquals(64, d.cacheHitSnapshot().reuseTokens());
        }
    }

    @Test void deployedMtpForwardBelongsToPrefillAndDecodeRetainsItsOwnStep() {
        var reports = new ArrayList<String>();
        var sink = (FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(getClass().getClassLoader(),
                new Class<?>[]{FlexMonitor.class}, (proxy, method, args) -> {
                    if (method.getName().equals("report"))
                        reports.add(((FlexMetricTags) args[1]).getTags().get("role") + ":" + args[0]);
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        var p = Map.of("role", "ROLE_TYPE_PREFILL");
        var d = Map.of("role", "ROLE_TYPE_DECODE");
        monitor.reportEvent(Map.of("rtp_llm_model_forward_us", 2000), p);
        monitor.reportEvent(Map.of("rtp_llm_model_forward_us", 30000), d);
        monitor.reportEvent(MockDecodeStepMetrics.values(30, 4, 1), d);
        monitor.reportEvent(MockDecodeStepMetrics.values(30, 4, 1), p);
        assertEquals(1, reports.stream().filter(x -> x.endsWith(":rtp_llm_model_forward_us")).count());
        assertTrue(reports.contains("ROLE_TYPE_PREFILL:rtp_llm_model_forward_us"));
        assertTrue(reports.contains("ROLE_TYPE_DECODE:mock_decode_step_forward_us"));
        assertFalse(reports.contains("ROLE_TYPE_PREFILL:mock_decode_step_forward_us"));
        assertFalse(reports.contains("ROLE_TYPE_PREFILL:rtp_llm_sp_avg_accept_token_num"));
        assertTrue(reports.contains("ROLE_TYPE_PREFILL:rtp_llm_sp_step_latency_us"),
                "MtpExecutor::process reports step latency for both roles");
        monitor.reportEvent(MockStreamMetrics.completed(128, 1, 64, 1000, 2000), p);
        monitor.reportEvent(MockStreamMetrics.completed(128, 8, 64, 3000, 9000), d);
        for (String name : MockStreamMetrics.completed(128, 8, 64, 3000, 9000).keySet()) {
            assertTrue(reports.contains("ROLE_TYPE_PREFILL:" + name), name);
            assertTrue(reports.contains("ROLE_TYPE_DECODE:" + name), name);
        }
        for (String name : new String[]{"mock_kv_cache_hit_rate", "rtp_llm_kv_cache_reuse_length",
                "rtp_llm_stream_cache_device_reuse_length", "rtp_llm_stream_cache_memory_reuse_length"}) {
            monitor.reportEvent(Map.of(name, 0), p);
            monitor.reportEvent(Map.of(name, 0), d);
            assertTrue(reports.contains("ROLE_TYPE_PREFILL:" + name), name);
            assertTrue(reports.contains("ROLE_TYPE_DECODE:" + name), name);
        }
    }

    @Test void idleDecodeDoesNotDiluteExecutionBatchAverageButStillReportsRunningZero() {
        var batches = new ArrayList<Double>();
        var running = new ArrayList<Double>();
        var sink = (FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(getClass().getClassLoader(),
                new Class<?>[]{FlexMonitor.class}, (proxy, method, args) -> {
                    if (method.getName().equals("report")) {
                        if (args[0].equals("rtp_llm_generate_batch_size")) batches.add(((Number) args[2]).doubleValue());
                        if (args[0].equals("rtp_llm_running_stream_size")) running.add(((Number) args[2]).doubleValue());
                    }
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        var d = Map.of("role", "ROLE_TYPE_DECODE");
        monitor.reportScheduler(Map.of("rtp_llm_generate_batch_size", 4, "rtp_llm_running_stream_size", 4), d);
        monitor.reportScheduler(Map.of("rtp_llm_generate_batch_size", 0, "rtp_llm_running_stream_size", 0), d);
        monitor.sample(Map.of("rtp_llm_generate_batch_size", 0), d, System.nanoTime());
        monitor.sample(Map.of("rtp_llm_generate_batch_size", 0), d, System.nanoTime());
        assertEquals(java.util.List.of(4.0), batches);
        assertEquals(java.util.List.of(4.0, 0.0), running);
    }

    @Test void streamDimensionsUseLocalOutputAndDoNotInventZeroGaugeSamples() {
        var p = MockStreamMetrics.completed(1024, 1, 1024, 2000, 2000);
        var d = MockStreamMetrics.completed(1024, 400, 0, 5000, 50000);
        assertEquals(1L, p.get("rtp_llm_output_token_length"));
        assertEquals(400L, d.get("rtp_llm_output_token_length"));
        assertFalse(p.containsKey("rtp_llm_effective_context_length"));
        assertFalse(d.containsKey("rtp_llm_reuse_length"));
        assertEquals(1024L, d.get("rtp_llm_effective_context_length"));
        assertEquals(5000L, d.get("rtp_llm_first_token_latency_us"));
        assertEquals(50000L, d.get("rtp_llm_latency_us"));
        assertTrue(MockStreamMetrics.completed(0, 0, 0, 0, 0).isEmpty());
        assertFalse(MockStreamMetrics.completed(128, 0, 0, 2000, 2000)
                .containsKey("rtp_llm_first_token_latency_us"));
    }
}
