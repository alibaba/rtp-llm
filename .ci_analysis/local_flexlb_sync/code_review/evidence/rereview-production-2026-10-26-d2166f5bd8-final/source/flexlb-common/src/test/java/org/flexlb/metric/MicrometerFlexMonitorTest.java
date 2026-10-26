package org.flexlb.metric;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import org.flexlb.config.MetricsCardinalityConfig;
import org.flexlb.enums.FlexMetricType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class MicrometerFlexMonitorTest {

    @Test
    void registrationGrowthKeepsConcurrentUpdatesAndScrapesVisible() throws Exception {
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try (var executor = Executors.newFixedThreadPool(3)) {
            registry.config().meterFilter(new MetricsCardinalityConfig().removeEngineTagsFilter());
            var monitor = new MicrometerFlexMonitor(registry);
            var first = FlexMetricTags.ofEngine("worker-0", "role", "PREFILL");
            monitor.prepare("worker.depth", first);
            var gauge = registry.get("flexlb.worker.depth").gauge();
            var start = new CountDownLatch(1);
            var register =
                    executor.submit(
                            () -> {
                                start.await();
                                for (int worker = 1; worker < 1500; worker++) {
                                    monitor.report(
                                            "worker.depth",
                                            FlexMetricTags.ofEngine(
                                                    "worker-" + worker, "role", "PREFILL"),
                                            1);
                                }
                                return null;
                            });
            var update =
                    executor.submit(
                            () -> {
                                start.await();
                                for (int i = 0; i <= 10000; i++)
                                    monitor.report("worker.depth", first, i);
                                return null;
                            });
            var scrape =
                    executor.submit(
                            () -> {
                                start.await();
                                for (int i = 0; i < 10000; i++) {
                                    double value = gauge.value();
                                    assertTrue(
                                            value >= 0 && value <= 11499,
                                            "invalid partial aggregate: " + value);
                                }
                                return null;
                            });
            start.countDown();
            register.get(10, TimeUnit.SECONDS);
            update.get(10, TimeUnit.SECONDS);
            scrape.get(10, TimeUnit.SECONDS);
            assertExportedDepth(registry, "PREFILL", 11499);
        } finally {
            registry.close();
        }
    }

    @Test
    void absentAndEmptyTagsAreTheSameGaugeSource() {
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            monitor.prepare("worker.depth", null);
            monitor.prepare("worker.depth", FlexMetricTags.of());
            monitor.report("worker.depth", 4);
            monitor.report("worker.depth", FlexMetricTags.of(), 6);
            assertEquals(6, registry.get("flexlb.worker.depth").gauge().value());
            monitor.report("worker.depth", 3);
            assertEquals(3, registry.get("flexlb.worker.depth").gauge().value());
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void collapsedGaugeExportsEveryWorkersLatestValue(boolean prepared) {
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try {
            registry.config().meterFilter(new MetricsCardinalityConfig().removeEngineTagsFilter());
            var monitor = new MicrometerFlexMonitor(registry);
            monitor.register("worker.depth", FlexMetricType.GAUGE);
            var first = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL");
            var second = FlexMetricTags.ofEngine("10.0.0.2", "role", "PREFILL");
            if (prepared) {
                monitor.prepare("worker.depth", first);
                monitor.prepare("worker.depth", second);
            }
            monitor.report("worker.depth", first, 4);
            monitor.report("worker.depth", second, 6);
            assertExportedDepth(registry, "PREFILL", 10);

            monitor.report("worker.depth", second, 3);
            assertExportedDepth(registry, "PREFILL", 7);
            monitor.prepare("worker.depth", second);
            monitor.report("worker.depth", second, 3);
            assertExportedDepth(registry, "PREFILL", 7);
            monitor.report("worker.depth", first, 0);
            assertExportedDepth(registry, "PREFILL", 3);

            monitor.report(
                    "worker.depth", FlexMetricTags.ofEngine("10.0.0.1", "role", "DECODE"), 8);
            monitor.report(
                    "worker.depth", FlexMetricTags.ofEngine("10.0.0.2", "role", "DECODE"), 9);
            assertExportedDepth(registry, "DECODE", 17);
            assertExportedDepth(registry, "PREFILL", 3);
            assertEquals(2, registry.find("flexlb.worker.depth").gauges().size());
        } finally {
            registry.close();
        }
    }

    @Test
    void perEngineGaugesRemainIndependentWithoutTheFilter() {
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var first = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL");
            var second = FlexMetricTags.ofEngine("10.0.0.2", "role", "PREFILL");
            monitor.report("worker.depth", first, 4);
            monitor.report("worker.depth", second, 6);
            monitor.report("worker.depth", second, 3);
            assertEquals(2, registry.find("flexlb.worker.depth").gauges().size());
            assertEquals(
                    4,
                    registry.get("flexlb.worker.depth")
                            .tag("engineIp", "10.0.0.1")
                            .gauge()
                            .value());
            assertEquals(
                    3,
                    registry.get("flexlb.worker.depth")
                            .tag("engineIp", "10.0.0.2")
                            .gauge()
                            .value());
        } finally {
            registry.close();
        }
    }

    @Test
    void concurrentFirstRegistrationsShareOneExportedGauge() throws Exception {
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try (var executor = Executors.newFixedThreadPool(8)) {
            registry.config().meterFilter(new MetricsCardinalityConfig().removeEngineTagsFilter());
            var monitor = new MicrometerFlexMonitor(registry);
            var start = new CountDownLatch(1);
            var results = new ArrayList<Future<?>>();
            for (int worker = 0; worker < 32; worker++) {
                int id = worker;
                results.add(
                        executor.submit(
                                () -> {
                                    start.await();
                                    var tags =
                                            FlexMetricTags.ofEngine(
                                                    "worker-" + id, "role", "PREFILL");
                                    for (int update = 0; update <= 100; update++) {
                                        monitor.report("worker.depth", tags, id + update);
                                    }
                                    return null;
                                }));
            }
            start.countDown();
            for (var result : results) {
                result.get(5, TimeUnit.SECONDS);
            }
            assertEquals(1, registry.find("flexlb.worker.depth").gauges().size());
            assertExportedDepth(registry, "PREFILL", 32 * 100 + 31 * 32 / 2);
        } finally {
            registry.close();
        }
    }

    @Test
    void replacingANonFiniteSourceRestoresTheAggregate() {
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try {
            registry.config().meterFilter(new MetricsCardinalityConfig().removeEngineTagsFilter());
            var monitor = new MicrometerFlexMonitor(registry);
            var first = FlexMetricTags.ofEngine("10.0.0.1", "role", "PREFILL");
            var second = FlexMetricTags.ofEngine("10.0.0.2", "role", "PREFILL");
            monitor.report("worker.depth", first, 4);
            monitor.report("worker.depth", second, Double.NaN);
            assertTrue(Double.isNaN(registry.get("flexlb.worker.depth").gauge().value()));
            monitor.report("worker.depth", second, 6);
            assertExportedDepth(registry, "PREFILL", 10);
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @EnumSource(
            value = FlexMetricType.class,
            names = {"COUNTER", "TIMER"})
    void collapsedCountersAndTimersStillAccumulateAllWorkers(FlexMetricType type) {
        var registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        try {
            registry.config().meterFilter(new MetricsCardinalityConfig().removeEngineTagsFilter());
            var monitor = new MicrometerFlexMonitor(registry);
            monitor.register("worker.activity", type);
            monitor.report("worker.activity", FlexMetricTags.ofEngine("10.0.0.1"), 4);
            monitor.report("worker.activity", FlexMetricTags.ofEngine("10.0.0.2"), 6);
            if (type == FlexMetricType.COUNTER) {
                assertEquals(10, registry.get("flexlb.worker.activity").counter().count());
            } else {
                var timer = registry.get("flexlb.worker.activity").timer();
                assertEquals(2, timer.count());
                assertEquals(10, timer.totalTime(TimeUnit.MILLISECONDS));
            }
        } finally {
            registry.close();
        }
    }

    private static void assertExportedDepth(
            PrometheusMeterRegistry registry, String role, double expected) {
        assertEquals(
                expected, registry.get("flexlb.worker.depth").tag("role", role).gauge().value());
        var samples =
                registry.scrape()
                        .lines()
                        .filter(
                                line ->
                                        line.startsWith("flexlb_worker_depth{")
                                                && line.contains("role=\"" + role + "\""))
                        .toList();
        assertEquals(1, samples.size());
        assertFalse(samples.getFirst().contains("engineIp="));
        assertTrue(samples.getFirst().endsWith(" " + expected), samples.getFirst());
    }

    @AfterEach
    void resetAllowedMetrics() {
        MicrometerFlexMonitor.setAllowedMetrics(null);
    }

    @ParameterizedTest
    @CsvSource({
        "GAUGE,false",
        "GAUGE,true",
        "COUNTER,false",
        "COUNTER,true",
        "TIMER,false",
        "TIMER,true"
    })
    void backendFailureDropsOnlyBrokenMetric(FlexMetricType type, boolean error) {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        try {
            registry.config()
                    .meterFilter(
                            new MeterFilter() {
                                @Override
                                public Meter.Id map(Meter.Id id) {
                                    if (id.getName().equals("flexlb.broken")) {
                                        if (error) {
                                            throw new AssertionError("monitor failed");
                                        }
                                        throw new IllegalStateException("monitor failed");
                                    }
                                    return id;
                                }
                            });
            var monitor = new MicrometerFlexMonitor(registry);
            monitor.register("broken", type);
            monitor.register("healthy", type);
            assertDoesNotThrow(() -> monitor.prepare("broken", null));
            assertDoesNotThrow(() -> monitor.report("broken", 3));
            monitor.report("healthy", 7);
            switch (type) {
                case GAUGE -> assertEquals(7, registry.get("flexlb.healthy").gauge().value());
                case COUNTER -> assertEquals(7, registry.get("flexlb.healthy").counter().count());
                case TIMER ->
                        assertEquals(
                                7,
                                registry.get("flexlb.healthy")
                                        .timer()
                                        .totalTime(TimeUnit.MILLISECONDS));
                default -> throw new AssertionError(type);
            }
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void brokenTagsDoNotEscapeReporting(boolean error) {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var tags = mock(FlexMetricTags.class);
            if (error) {
                when(tags.isEmpty()).thenThrow(new AssertionError("bad tags"));
            } else {
                when(tags.isEmpty()).thenThrow(new IllegalStateException("bad tags"));
            }
            assertDoesNotThrow(() -> monitor.prepare("broken.tags", tags));
            assertDoesNotThrow(() -> monitor.report("broken.tags", tags, 1));
            monitor.report("healthy", 2);
            assertEquals(2, registry.get("flexlb.healthy").gauge().value());
        } finally {
            registry.close();
        }
    }

    @Test
    void invalidRegistrationAndLookupDoNotEscape() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            assertDoesNotThrow(() -> monitor.register(null, FlexMetricType.GAUGE));
            assertDoesNotThrow(() -> monitor.register("invalid", null));
            assertDoesNotThrow(() -> monitor.prepare(null, null));
            assertDoesNotThrow(() -> monitor.report(null, 1));
            monitor.report("healthy", 2);
            assertEquals(2, registry.get("flexlb.healthy").gauge().value());
        } finally {
            registry.close();
        }
    }

    @Test
    void prepareTimerRegistersWithoutRecordingAndReportReusesIt() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            FlexMetricTags tags = FlexMetricTags.of("role", "PREFILL", "engineIp", "127.0.0.1");
            monitor.register("test.timer", FlexMetricType.TIMER);

            monitor.prepare("test.timer", tags);

            Timer prepared =
                    registry.find("flexlb.test.timer")
                            .tags("role", "PREFILL", "engineIp", "127.0.0.1")
                            .timer();
            assertNotNull(prepared);
            assertEquals(0L, prepared.count());

            monitor.report("test.timer", tags, 12.0D);

            Timer reported =
                    registry.find("flexlb.test.timer")
                            .tags("role", "PREFILL", "engineIp", "127.0.0.1")
                            .timer();
            assertEquals(prepared, reported);
            assertEquals(1L, reported.count());
            assertEquals(12.0D, reported.totalTime(TimeUnit.MILLISECONDS), 0.001D);
        } finally {
            monitor.close();
            registry.close();
        }
    }
}
