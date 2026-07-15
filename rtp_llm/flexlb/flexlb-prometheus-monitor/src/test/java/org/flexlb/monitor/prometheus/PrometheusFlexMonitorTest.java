package org.flexlb.monitor.prometheus;

import io.prometheus.client.Collector;
import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.Gauge;
import io.micrometer.core.instrument.Clock;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexStatisticsType;
import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrometheusFlexMonitorTest {

    @Test
    void counterNormalizesNamePreservesTagsAndAccumulatesIncrements() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        monitor.register(
                "grpc.call.count",
                FlexMetricType.QPS,
                FlexMetricTags.of("method", "", "status", ""));

        FlexMetricTags tags = FlexMetricTags.of("method", "chat/completions", "status", "ok");
        monitor.report("grpc.call.count", tags, 2.0);
        monitor.report("grpc.call.count", tags, 3.0);

        assertEquals(5.0, registry.getSampleValue(
                "flexlb_grpc_call_count_total",
                new String[]{"method", "status"},
                new String[]{"chat/completions", "ok"}));
    }

    @Test
    void registerCreatesUnlabelledBindingBeforeFirstReport() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);

        monitor.register("registered.gauge", FlexMetricType.GAUGE, FlexMetricTags.of());

        assertEquals(0.0, registry.getSampleValue("flexlb_registered_gauge"));
    }

    @Test
    void repeatedEquivalentRegistrationPreservesCounterBinding() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        monitor.register("repeated.registration", FlexMetricType.COUNTER, FlexMetricTags.of());

        monitor.report("repeated.registration", 1.0);
        monitor.register("repeated.registration", FlexMetricType.COUNTER, FlexMetricTags.of());
        monitor.report("repeated.registration", 2.0);

        assertEquals(3.0, registry.getSampleValue("flexlb_repeated_registration_total"));
    }

    @Test
    void gaugeSetsLatestValueAndKeepsExactlyOnePrefix() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        monitor.register("flexlb.queue-depth", FlexMetricType.GAUGE, FlexMetricTags.of());

        monitor.report("flexlb.queue-depth", 8.0);
        monitor.report("flexlb.queue-depth", 3.0);

        assertEquals(3.0, registry.getSampleValue("flexlb_queue_depth"));
    }

    @Test
    void summaryExportsCountSumAndConfiguredQuantiles() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        int statistics = FlexStatisticsType.SUM | FlexStatisticsType.SUMMARY
                | FlexStatisticsType.MIN | FlexStatisticsType.MAX
                | FlexStatisticsType.PERCENTILE_75 | FlexStatisticsType.PERCENTILE_95
                | FlexStatisticsType.PERCENTILE_99;
        monitor.register("request.latency", FlexMetricType.GAUGE, statistics, FlexMetricTags.of());

        monitor.report("request.latency", 10.0);
        monitor.report("request.latency", 20.0);

        assertEquals(2.0, registry.getSampleValue("flexlb_request_latency_count"));
        assertEquals(30.0, registry.getSampleValue("flexlb_request_latency_sum"));
        assertEquals(10.0, quantile(registry, "flexlb_request_latency", "0.0"));
        assertEquals(20.0, quantile(registry, "flexlb_request_latency", "1.0"));
        assertEquals(20.0, quantile(registry, "flexlb_request_latency", "0.75"));
        assertEquals(20.0, quantile(registry, "flexlb_request_latency", "0.95"));
        assertEquals(20.0, quantile(registry, "flexlb_request_latency", "0.99"));
    }

    @Test
    void statisticsOverloadWithZeroMaskStillCreatesSummary() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        monitor.register("payload.size", FlexMetricType.GAUGE, 0, FlexMetricTags.of());

        monitor.report("payload.size", 12.0);

        assertEquals(1.0, registry.getSampleValue("flexlb_payload_size_count"));
        assertEquals(12.0, registry.getSampleValue("flexlb_payload_size_sum"));
    }

    @Test
    void sortsTagKeysAndDropsReportsWithChangedSchema() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        Map<String, String> reverseOrder = new LinkedHashMap<>();
        reverseOrder.put("zone", "east");
        reverseOrder.put("method", "chat");
        monitor.register("requests", FlexMetricType.COUNTER, FlexMetricTags.of(reverseOrder));

        monitor.report("requests", FlexMetricTags.of(reverseOrder), 1.0);
        assertDoesNotThrow(() -> monitor.report(
                "requests", FlexMetricTags.of("method", "chat"), 10.0));

        assertEquals(1.0, registry.getSampleValue(
                "flexlb_requests_total",
                new String[]{"method", "zone"},
                new String[]{"chat", "east"}));
    }

    @Test
    void unregisteredMetricIsDroppedWithoutCreatingCollector() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);

        assertDoesNotThrow(() -> monitor.report("active.workers", 4.0));

        assertNull(registry.getSampleValue("flexlb_active_workers"));
    }

    @Test
    void invalidLabelNameDoesNotEscape() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);

        assertDoesNotThrow(() -> monitor.register(
                "invalid.labels", FlexMetricType.GAUGE, FlexMetricTags.of("not-a-label", "value")));
        assertDoesNotThrow(() -> monitor.report(
                "invalid.labels", FlexMetricTags.of("not-a-label", "value"), 1.0));
    }

    @Test
    void collectorRegistrationFailureDisablesLaterRetries() {
        CollectorRegistry registry = new CollectorRegistry();
        Gauge collision = Gauge.build()
                .name("flexlb_collision")
                .help("existing")
                .register(registry);
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        assertDoesNotThrow(() -> monitor.register("collision", FlexMetricType.GAUGE, FlexMetricTags.of()));

        registry.unregister(collision);
        assertDoesNotThrow(() -> monitor.report("collision", 2.0));

        assertEquals(null, registry.getSampleValue("flexlb_collision"));
    }

    @Test
    void negativeCounterUpdateDoesNotEscapeOrPreventLaterUpdates() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        monitor.register("events", FlexMetricType.COUNTER, FlexMetricTags.of());

        assertDoesNotThrow(() -> monitor.report("events", -1.0));
        monitor.report("events", 2.0);

        assertEquals(2.0, registry.getSampleValue("flexlb_events_total"));
    }

    @Test
    void concurrentFirstReportsRegisterOnceAndKeepEveryIncrement() throws Exception {
        SlowCollectorRegistry registry = new SlowCollectorRegistry();
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        monitor.register("concurrent.events", FlexMetricType.COUNTER, FlexMetricTags.of());
        int reports = 32;
        ExecutorService executor = Executors.newFixedThreadPool(reports);
        CountDownLatch start = new CountDownLatch(1);
        try {
            for (int i = 0; i < reports; i++) {
                executor.submit(() -> {
                    start.await();
                    monitor.report("concurrent.events", 1.0);
                    return null;
                });
            }
            start.countDown();
            executor.shutdown();
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
        } finally {
            executor.shutdownNow();
        }

        assertEquals(1, registry.registrations.get());
        assertEquals(32.0, registry.getSampleValue("flexlb_concurrent_events_total"));
    }

    @Test
    void priorityRegistrationAppearsInSharedMicrometerScrapeAsCounter() {
        CollectorRegistry registry = new CollectorRegistry();
        PrometheusMeterRegistry meterRegistry = new PrometheusMeterRegistry(
                PrometheusConfig.DEFAULT, registry, Clock.SYSTEM);
        PrometheusFlexMonitor monitor = new PrometheusFlexMonitor(registry);
        monitor.register(
                "routing.decisions",
                FlexMetricType.COUNTER,
                FlexPriorityType.CRITICAL,
                FlexMetricTags.of("result", ""));

        monitor.report("routing.decisions", FlexMetricTags.of("result", "selected"), 1.0);

        String scrape = meterRegistry.scrape();
        assertTrue(scrape.contains("# TYPE flexlb_routing_decisions_total counter"), scrape);
        assertTrue(scrape.contains("flexlb_routing_decisions_total{result=\"selected\",} 1.0"), scrape);
    }

    private static final class SlowCollectorRegistry extends CollectorRegistry {
        private final AtomicInteger registrations = new AtomicInteger();

        @Override
        public void register(Collector collector) {
            registrations.incrementAndGet();
            try {
                Thread.sleep(20);
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(error);
            }
            super.register(collector);
        }
    }

    private static Double quantile(CollectorRegistry registry, String name, String quantile) {
        return registry.getSampleValue(name, new String[]{"quantile"}, new String[]{quantile});
    }
}
