package org.flexlb.mock.metrics;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import io.prometheus.client.Collector.MetricFamilySamples;

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
}
