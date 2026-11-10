package org.flexlb.mock.metrics;

import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.flexlb.mock.metrics.MetricReporter.EventKind;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies explicit event types and sample weights across scrapes and push reporting.
 */
class EventMetricsTest {
    @Test void summariesKeepLabelsAndDoNotExposeAmbiguousWindowMeans() {
        var registry = new CollectorRegistry();
        var metrics = prometheus("prefill", registry);
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

    @Test void repeatedAndIdleScrapesNeverConsumeEvents() {
        var registry = new CollectorRegistry();
        var metrics = prometheus("vit", registry);
        metrics.observe("py_rtp_vit_embedding_rt", 10); metrics.observe("py_rtp_vit_embedding_rt", 30);
        metrics.increment("py_rtp_vit_qps"); metrics.increment("py_rtp_vit_qps");
        metrics.add("embedding_tokens", 20); metrics.add("embedding_tokens", 40);
        metrics.observe("sample_qps", 4);
        assertThrows(IllegalArgumentException.class, () -> metrics.add("sample_qps", 1));
        assertThrows(IllegalArgumentException.class, () -> metrics.observe("embedding_tokens", 1));
        for (int i = 0; i < 3; i++) {
            var text = MetricTags.scrape(registry, Map.of());
            assertEquals(2, sample(text, "py_rtp_vit_qps_total", ""));
            assertEquals(60, sample(text, "embedding_tokens_total", ""));
            assertEquals(4, sample(text, "sample_qps_sum", ""));
            assertEquals(40, sample(text, "py_rtp_vit_embedding_rt_sum", ""));
            assertEquals(2, sample(text, "py_rtp_vit_embedding_rt_count", ""));
            assertTrue(text.contains("# TYPE py_rtp_vit_qps_total counter"));
            assertFalse(text.contains("py_rtp_vit_qps{"));
        }
        metrics.increment("py_rtp_vit_qps");
        assertEquals(3, sample(MetricTags.scrape(registry, Map.of()), "py_rtp_vit_qps_total", ""));
        assertThrows(IllegalArgumentException.class, () -> metrics.observe("x", Double.NaN));
    }

    @Test void unequalInstancesAndRepeatedScrapesPreserveAll101RequestWeights() {
        var firstRegistry = new CollectorRegistry(); var secondRegistry = new CollectorRegistry();
        var first = prometheus("prefill", firstRegistry);
        var second = prometheus("prefill", secondRegistry);
        for (int i = 0; i < 100; i++) first.observe("input_length", 10);
        second.observe("input_length", 1000);
        var a = MetricTags.scrape(firstRegistry, Map.of("instance", "a"));
        var b = MetricTags.scrape(secondRegistry, Map.of("instance", "b"));
        double total = sample(a, "input_length_sum", "") + sample(b, "input_length_sum", "");
        double count = sample(a, "input_length_count", "") + sample(b, "input_length_count", "");
        assertEquals(101, count); assertEquals(2000.0 / 101, total / count);
        assertNotEquals(505, total / count);
        // A new scrape period does not turn a single new request into the weight of 100 old ones.
        first.observe("input_length", 1000);
        a = MetricTags.scrape(firstRegistry, Map.of());
        assertEquals(2000, sample(a, "input_length_sum", ""));
        assertEquals(101, sample(a, "input_length_count", ""));
    }

    @Test void sdkReceivesOriginalSamplesAndNoPeriodicEventFacade() {
        record Event(String name, EventKind kind, Map<String, String> tags, double value) {}
        var received = new ArrayList<Event>();
        MetricReporter reporter = new MetricReporter() {
            public void initialize(Map<String,String> env) { }
            public void observe(String name, EventKind kind, Map<String,String> tags, double value) { received.add(new Event(name, kind, tags, value)); }
            public void report(List<MetricFamilySamples> snapshot) { assertTrue(snapshot.isEmpty()); }
        };
        var events = new EventMetrics("prefill", reporter, Map.of("dp_rank", "0"));
        var registry = new CollectorRegistry(); events.register(registry);
        for (int i = 0; i < 100; i++) events.observe("input_length", 10);
        events.observe("input_length", 1000);
        events.observe("reuse_length", 0, Map.of("tier", "DEVICE"));
        events.increment("streaming_qps");
        assertEquals(103, received.size());
        assertEquals(10, received.getFirst().value()); assertEquals(1000, received.get(100).value());
        assertEquals(Map.of("role", "prefill", "dp_rank", "0", "tier", "DEVICE"), received.get(101).tags());
        assertEquals(0, received.get(101).value()); assertEquals(1, received.getLast().value());
        assertEquals(EventKind.SAMPLE, received.get(101).kind());
        assertEquals(EventKind.DELTA, received.getLast().kind());
        reporter.report(MetricTags.collect(registry, Map.of()));
        assertEquals(103, received.size());
    }

    private static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }

    private static EventMetrics prometheus(String role, CollectorRegistry registry) {
        var metrics = new EventMetrics(role, null, Map.of()); metrics.register(registry); return metrics;
    }
}
