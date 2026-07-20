package org.flexlb.metric;

import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MicrometerFlexMonitorTest {

    @AfterEach
    void resetAllowedMetrics() {
        MicrometerFlexMonitor.setAllowedMetrics(null);
    }

    @Test
    void prepareTimerRegistersWithoutRecordingAndReportReusesIt() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            FlexMetricTags tags = FlexMetricTags.of("role", "PREFILL", "engineIp", "127.0.0.1");
            monitor.register("test.timer", FlexMetricType.TIMER);

            monitor.prepare("test.timer", tags);

            Timer prepared = registry.find("flexlb.test.timer")
                    .tags("role", "PREFILL", "engineIp", "127.0.0.1")
                    .timer();
            assertNotNull(prepared);
            assertEquals(0L, prepared.count());

            monitor.report("test.timer", tags, 12.0D);

            Timer reported = registry.find("flexlb.test.timer")
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

    @Test
    void endpointScopeRemovesEveryTaggedMeterAndLocalCache() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry, TimeUnit.DAYS.toMillis(1));
        try {
            FlexMetricTags endpointTags = endpointTags("127.0.0.1:8001");
            FlexMetricTags reasonTags = FlexMetricTags.ofEngine(
                    "127.0.0.1", "127.0.0.1:8001", "role", "PREFILL", "reason", "batch_full");
            monitor.register("test.timer", FlexMetricType.TIMER);
            monitor.register("test.counter", FlexMetricType.QPS);
            monitor.register("test.gauge", FlexMetricType.GAUGE);

            MetricLease lease = monitor.acquireEndpointScope(endpointTags, 0L);
            monitor.report("test.timer", endpointTags, 7.0D);
            monitor.report("test.counter", reasonTags, 1.0D);
            monitor.report("test.gauge", endpointTags, 3.0D);
            assertEquals(3, monitor.cacheStats().meters());
            assertNotNull(registry.find("flexlb.test.timer").timer());
            assertNotNull(registry.find("flexlb.test.counter").counter());
            assertNotNull(registry.find("flexlb.test.gauge").gauge());

            lease.close();
            monitor.runRetirementSweep();

            assertTrue(registry.getMeters().isEmpty());
            assertEquals(0, monitor.cacheStats().meters());
            assertEquals(0, monitor.cacheStats().tags());
            assertEquals(1, monitor.cacheStats().endpointScopes());

            // A callback that arrives after hard retirement must not resurrect the meter.
            monitor.report("test.timer", endpointTags, 9.0D);
            assertNull(registry.find("flexlb.test.timer").timer());

            monitor.runRetirementSweep();
            assertEquals(0, monitor.cacheStats().endpointScopes());
        } finally {
            monitor.close();
            registry.close();
        }
    }

    @Test
    void endpointScopeKeepsMetersDuringGracePeriod() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry, TimeUnit.DAYS.toMillis(1));
        try {
            FlexMetricTags tags = endpointTags("127.0.0.1:8002");
            monitor.register("test.timer", FlexMetricType.TIMER);
            MetricLease lease = monitor.acquireEndpointScope(tags, 60_000L);
            monitor.report("test.timer", tags, 1.0D);

            lease.close();
            monitor.runRetirementSweep();
            monitor.report("test.timer", tags, 2.0D);

            Timer timer = registry.find("flexlb.test.timer").timer();
            assertNotNull(timer);
            assertEquals(2L, timer.count());
        } finally {
            monitor.close();
            registry.close();
        }
    }

    @Test
    void quickReaddPreventsOldLeaseFromRemovingNewEndpointMetrics() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry, TimeUnit.DAYS.toMillis(1));
        try {
            FlexMetricTags tags = endpointTags("127.0.0.1:8003");
            monitor.register("test.counter", FlexMetricType.QPS);
            MetricLease oldLease = monitor.acquireEndpointScope(tags, 0L);
            monitor.report("test.counter", tags, 1.0D);
            oldLease.close();

            MetricLease newLease = monitor.acquireEndpointScope(tags, 0L);
            monitor.runRetirementSweep();
            assertNotNull(registry.find("flexlb.test.counter").counter());
            assertEquals(1, monitor.cacheStats().endpointScopes());

            oldLease.close();
            newLease.close();
            monitor.runRetirementSweep();
            assertNull(registry.find("flexlb.test.counter").counter());
        } finally {
            monitor.close();
            registry.close();
        }
    }

    @Test
    void retiringOneEndpointDoesNotRemoveAnotherEndpointsMeters() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry, TimeUnit.DAYS.toMillis(1));
        try {
            FlexMetricTags firstTags = endpointTags("127.0.0.1:8005");
            FlexMetricTags secondTags = endpointTags("127.0.0.1:8006");
            monitor.register("test.counter", FlexMetricType.QPS);
            MetricLease firstLease = monitor.acquireEndpointScope(firstTags, 0L);
            MetricLease secondLease = monitor.acquireEndpointScope(secondTags, 0L);
            monitor.report("test.counter", firstTags, 1.0D);
            monitor.report("test.counter", secondTags, 1.0D);

            firstLease.close();
            monitor.runRetirementSweep();

            assertNull(registry.find("flexlb.test.counter")
                    .tag("engineIpPort", "127.0.0.1:8005").counter());
            assertNotNull(registry.find("flexlb.test.counter")
                    .tag("engineIpPort", "127.0.0.1:8006").counter());
            assertEquals(1, monitor.cacheStats().meters());
            secondLease.close();
        } finally {
            monitor.close();
            registry.close();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void concurrentLeasesDoNotRetireAnEndpointStillInUse() throws Exception {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry, TimeUnit.DAYS.toMillis(1));
        ExecutorService executor = Executors.newFixedThreadPool(16);
        try {
            FlexMetricTags tags = endpointTags("127.0.0.1:8004");
            monitor.register("test.counter", FlexMetricType.QPS);
            MetricLease owner = monitor.acquireEndpointScope(tags, 0L);
            monitor.report("test.counter", tags, 1.0D);
            CountDownLatch start = new CountDownLatch(1);
            List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
            for (int thread = 0; thread < 16; thread++) {
                futures.add(executor.submit(() -> {
                    start.await();
                    for (int i = 0; i < 1_000; i++) {
                        MetricLease candidate = monitor.acquireEndpointScope(tags, 0L);
                        candidate.close();
                        candidate.close();
                    }
                    return null;
                }));
            }
            start.countDown();
            for (java.util.concurrent.Future<?> future : futures) {
                future.get();
            }

            monitor.runRetirementSweep();
            assertNotNull(registry.find("flexlb.test.counter").counter());
            assertEquals(1, monitor.cacheStats().endpointScopes());

            owner.close();
            monitor.runRetirementSweep();
            assertNull(registry.find("flexlb.test.counter").counter());
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            monitor.close();
            registry.close();
        }
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void uniqueEndpointChurnRemainsBounded() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry, TimeUnit.DAYS.toMillis(1));
        try {
            monitor.register("test.counter", FlexMetricType.QPS);
            int endpointCount = 100_000;
            int batchSize = 250;
            for (int i = 0; i < endpointCount; i++) {
                FlexMetricTags tags = endpointTags("127.0.0.1:" + (10_000 + i));
                MetricLease lease = monitor.acquireEndpointScope(tags, 0L);
                monitor.report("test.counter", tags, 1.0D);
                lease.close();
                if ((i + 1) % batchSize == 0) {
                    monitor.runRetirementSweep();
                    monitor.runRetirementSweep();
                    assertEquals(0, monitor.cacheStats().meters());
                    assertTrue(monitor.cacheStats().endpointScopes() <= batchSize);
                }
            }

            assertTrue(registry.getMeters().isEmpty());
            assertEquals(0, monitor.cacheStats().meters());
            assertEquals(0, monitor.cacheStats().tags());
            assertEquals(0, monitor.cacheStats().endpointScopes());
        } finally {
            monitor.close();
            registry.close();
        }
    }

    @Test
    void timerDoesNotPublishHistogramBuckets() {
        // Histogram buckets are hardcoded to false; only quantile gauges should be published.
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            monitor.register("test.timer", FlexMetricType.TIMER);
            monitor.report("test.timer", null, 5.0);

            // Timer is registered and records the value
            Timer timer = registry.find("flexlb.test.timer").timer();
            assertNotNull(timer, "Timer should be registered");
            assertEquals(1L, timer.count());

            // Quantile gauges should still be present
            assertNotNull(registry.find("flexlb.test.timer.percentile").tag("phi", "0.5").gauge(),
                    "P50 percentile gauge should exist");
            assertNotNull(registry.find("flexlb.test.timer.percentile").tag("phi", "0.99").gauge(),
                    "P99 percentile gauge should exist");

            // Histogram bucket gauges should NOT be present
            assertNull(registry.find("flexlb.test.timer.histogram").gauge(),
                    "Histogram bucket gauges should not exist");
        } finally {
            monitor.close();
            registry.close();
        }
    }

    private static FlexMetricTags endpointTags(String ipPort) {
        return FlexMetricTags.ofEngine("127.0.0.1", ipPort, "role", "PREFILL");
    }
}
