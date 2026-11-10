package org.flexlb.mock.metrics;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.flexlb.mock.metrics.MetricReporter.EventKind;
import io.prometheus.client.Collector.MetricFamilySamples;

/**
 * Exercises backend selection and ordered asynchronous metric delivery.
 */
class MetricsOutputTest {
    @Test void absentProviderLeavesCollectionExclusivelyToPrometheus() {
        assertNull(MetricsOutput.discover(List.of(), Map.of()));
        var collections = new AtomicInteger();
        try (var output = new MetricsOutput(null, () -> { collections.incrementAndGet(); return List.of(); })) {
            assertTrue(output.prometheus());
        }
        assertEquals(0, collections.get());
    }

    @Test void duplicateOrBrokenProvidersCannotSilentlyFallBack() {
        MetricReporter broken = new MetricReporter() {
            public void observe(String name, EventKind kind, Map<String,String> tags, double value) { fail("No event expected"); }
            public void initialize(Map<String,String> env) { throw new IllegalStateException("SDK unavailable"); }
            public void report(List<MetricFamilySamples> snapshot) { fail(); }
        };
        assertThrows(IllegalStateException.class, () -> MetricsOutput.discover(List.of(broken, broken), Map.of()));
        assertThrows(IllegalStateException.class, () -> MetricsOutput.discover(List.of(broken), Map.of()));
    }

    @Test void installedProviderOwnsSamplingAndShutdownStopsItsReader() throws Exception {
        var received = new CountDownLatch(2);
        var collections = new AtomicInteger();
        MetricReporter reporter = new MetricReporter() {
            public void observe(String name, EventKind kind, Map<String,String> tags, double value) { fail("No event expected"); }
            public void initialize(Map<String,String> env) { assertEquals("tags", env.get("kmonitorTags")); }
            public long intervalMillis() { return 10; }
            public void report(List<MetricFamilySamples> snapshot) { received.countDown(); }
        };
        reporter = MetricsOutput.discover(List.of(reporter), Map.of("kmonitorTags", "tags"));
        var output = new MetricsOutput(reporter, () -> { collections.incrementAndGet(); return List.of(); });
        try {
            assertFalse(output.prometheus());
            assertTrue(received.await(2, TimeUnit.SECONDS));
        } finally { output.close(); }
        int stopped = collections.get();
        Thread.sleep(40);
        assertEquals(stopped, collections.get());
    }
    @Test void slowSdkDoesNotBlockBusinessAndPreservesOriginalSampleOrder() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        var values = new ArrayList<Double>();
        MetricReporter sdk = new MetricReporter() {
            public void initialize(Map<String,String> env) { }
            public void observe(String name, EventKind kind, Map<String,String> tags, double value) {
                assertEquals(Map.of("role", "prefill"), tags);
                assertEquals(name.equals("tokens") ? EventKind.DELTA : EventKind.SAMPLE, kind);
                entered.countDown();
                try { assertTrue(release.await(2, TimeUnit.SECONDS)); }
                catch (InterruptedException error) { throw new RuntimeException(error); }
                values.add(value);
            }
            public void report(List<MetricFamilySamples> snapshot) { values.add(-1.0); }
        };
        var sender = new MetricsOutput.Sender(sdk, 128);
        try {
            var events = new EventMetrics("prefill", sender, Map.of());
            events.observe("input_length", 10);
            assertTrue(entered.await(1, TimeUnit.SECONDS));
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(1), () -> {
                for (int i = 0; i < 99; i++) events.observe("input_length", 10);
                events.observe("input_length", 1000);
                events.add("tokens", 17);
                sender.report(List.of());
            });
            release.countDown(); sender.close();
            assertEquals(103, values.size());
            assertEquals(2000, values.subList(0, 101).stream().mapToDouble(Double::doubleValue).sum());
            assertEquals(1000, values.get(100)); assertEquals(17, values.get(101)); assertEquals(-1, values.get(102));
        } finally { release.countDown(); sender.close(); }
    }

    @Test void overflowNeverBackpressuresBusinessAndIsAnExplicitIncompleteMetricStream() throws Exception {
        var entered = new CountDownLatch(1); var release = new CountDownLatch(1);
        MetricReporter sdk = new MetricReporter() {
            public void initialize(Map<String,String> env) { }
            public void observe(String name, EventKind kind, Map<String,String> tags, double value) {
                entered.countDown();
                try { release.await(); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
            }
            public void report(List<MetricFamilySamples> snapshot) { }
        };
        var sender = new MetricsOutput.Sender(sdk, 1);
        try {
            sender.observe("length", EventKind.SAMPLE, Map.of(), 1); assertTrue(entered.await(1, TimeUnit.SECONDS));
            sender.observe("length", EventKind.SAMPLE, Map.of(), 2);
            assertTimeoutPreemptively(java.time.Duration.ofSeconds(1), () -> sender.observe("length", EventKind.SAMPLE, Map.of(), 3));
            assertThrows(IllegalStateException.class, sender::checkHealthy);
        } finally {
            release.countDown(); assertThrows(IllegalStateException.class, sender::close);
        }
    }

}
