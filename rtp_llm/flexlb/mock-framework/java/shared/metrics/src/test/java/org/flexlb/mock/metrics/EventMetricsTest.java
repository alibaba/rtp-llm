package org.flexlb.mock.metrics;

import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class EventMetricsTest {
    private static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }
    private static EventMetrics prometheus(String role, AtomicLong clock, CollectorRegistry registry) {
        var metrics = new EventMetrics(role, clock::get, null, Map.of()); metrics.register(registry); return metrics;
    }

    @Test void summariesKeepLabelsAndDoNotExposeAmbiguousWindowMeans() {
        var clock = new AtomicLong(); var registry = new CollectorRegistry();
        var metrics = prometheus("prefill", clock, registry);
        var loaded = Map.of("load_attempted", "true"); var local = Map.of("load_attempted", "false");
        metrics.observe("latency_us", 10, loaded); metrics.observe("latency_us", 30, loaded);
        metrics.observe("latency_us", 5, local);
        var text = MetricTags.scrape(registry, Map.of());
        assertEquals(40, sample(text, "latency_us_sum", "load_attempted=\"true\""));
        assertEquals(2, sample(text, "latency_us_count", "load_attempted=\"true\""));
        assertEquals(5, sample(text, "latency_us_sum", "load_attempted=\"false\""));
        assertFalse(text.contains("latency_us{"));
        assertTrue(text.contains("# TYPE latency_us summary"));
        assertThrows(IllegalArgumentException.class, () -> metrics.observe("latency_us", 1));
    }

    @Test void idleScrapesRetainSumsAndCountsWhileQpsUsesWallTime() {
        var clock = new AtomicLong(); var registry = new CollectorRegistry();
        var metrics = prometheus("vit", clock, registry);
        metrics.observe("py_rtp_vit_embedding_rt", 10); metrics.observe("py_rtp_vit_embedding_rt", 30);
        metrics.increment("py_rtp_vit_qps"); metrics.increment("py_rtp_vit_qps");
        clock.set(2_000_000_000L);
        var text = MetricTags.scrape(registry, Map.of());
        assertEquals(1, sample(text, "py_rtp_vit_qps", ""));
        assertEquals(40, sample(text, "py_rtp_vit_embedding_rt_sum", ""));
        clock.set(3_000_000_000L); text = MetricTags.scrape(registry, Map.of());
        assertEquals(0, sample(text, "py_rtp_vit_qps", ""));
        assertEquals(40, sample(text, "py_rtp_vit_embedding_rt_sum", ""));
        assertEquals(2, sample(text, "py_rtp_vit_embedding_rt_count", ""));
        assertThrows(IllegalArgumentException.class, () -> metrics.observe("x", Double.NaN));
    }

    @Test void unequalInstancesAndUnequalScrapeWindowsPreserveAll101RequestWeights() {
        var clock = new AtomicLong();
        var firstRegistry = new CollectorRegistry(); var secondRegistry = new CollectorRegistry();
        var first = prometheus("prefill", clock, firstRegistry);
        var second = prometheus("prefill", clock, secondRegistry);
        for (int i = 0; i < 100; i++) first.observe("input_length", 10);
        second.observe("input_length", 1000);
        var a = MetricTags.scrape(firstRegistry, Map.of("instance", "a"));
        var b = MetricTags.scrape(secondRegistry, Map.of("instance", "b"));
        double total = sample(a, "input_length_sum", "") + sample(b, "input_length_sum", "");
        double count = sample(a, "input_length_count", "") + sample(b, "input_length_count", "");
        assertEquals(101, count); assertEquals(2000.0 / 101, total / count);
        assertNotEquals(505, total / count);
        // A new scrape period does not turn a single new request into the weight of 100 old ones.
        clock.set(1_000_000_000L); first.observe("input_length", 1000);
        a = MetricTags.scrape(firstRegistry, Map.of());
        assertEquals(2000, sample(a, "input_length_sum", ""));
        assertEquals(101, sample(a, "input_length_count", ""));
    }

    @Test void sdkReceivesOriginalSamplesAndNoPeriodicEventFacade() {
        record Event(String name, Map<String, String> tags, double value) {}
        var received = new ArrayList<Event>();
        MetricReporter reporter = new MetricReporter() {
            public void initialize(Map<String,String> env) { }
            public void observe(String name, Map<String,String> tags, double value) { received.add(new Event(name, tags, value)); }
            public void report(List<MetricFamilySamples> snapshot) { assertTrue(snapshot.isEmpty()); }
        };
        var events = new EventMetrics("prefill", System::nanoTime, reporter, Map.of("dp_rank", "0"));
        var registry = new CollectorRegistry(); events.register(registry);
        for (int i = 0; i < 100; i++) events.observe("input_length", 10);
        events.observe("input_length", 1000);
        events.observe("reuse_length", 0, Map.of("tier", "DEVICE"));
        events.increment("streaming_qps");
        assertEquals(103, received.size());
        assertEquals(10, received.getFirst().value()); assertEquals(1000, received.get(100).value());
        assertEquals(Map.of("role", "prefill", "dp_rank", "0", "tier", "DEVICE"), received.get(101).tags());
        assertEquals(0, received.get(101).value()); assertEquals(1, received.getLast().value());
        reporter.report(MetricTags.collect(registry, Map.of()));
        assertEquals(103, received.size());
    }
    private static double mean(String text, String name, String label) {
        return sample(text, name + "_sum", label) / sample(text, name + "_count", label);
    }
}
