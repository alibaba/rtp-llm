package org.flexlb.mock.metrics;

import static org.junit.jupiter.api.Assertions.*;
import io.prometheus.client.*;
import java.util.Map;
import org.junit.jupiter.api.Test;

class MetricTagsTest {
    @Test void followsRealWhaleEnvironmentWithoutInventingAddresses() {
        var tags = MetricTags.fromEnvironment(Map.of("HIPPO_APP", "app", "HIPPO_ROLE", "role",
            "HIPPO_SERVICE_NAME", "group", "HIPPO_SLAVE_IP", "10.0.0.1", "RequestedIP", "10.0.1.1"));
        assertEquals(Map.of("hippo_app", "app", "hippo_role", "role", "hippo_group", "group",
            "host_ip", "10.0.0.1", "container_ip", "10.0.1.1"), tags);
        assertTrue(MetricTags.fromEnvironment(Map.of()).isEmpty());
        assertEquals("10.0.0.1", MetricTags.fromEnvironment(Map.of("HIPPO_ROLE", "role", "HIPPO_SLAVE_IP", "10.0.0.1")).get("container_ip"));
    }

    @Test void enrichesCountersGaugesAndHistogramBucketsWithoutLosingDimensions() {
        var registry = new CollectorRegistry();
        Counter.build().name("calls").help("calls").labelNames("role").register(registry).labels("prefill").inc(2);
        Gauge.build().name("queue").help("queue").register(registry).set(3);
        Histogram.build().name("elapsed_seconds").help("elapsed").register(registry).observe(.1);
        String text = MetricTags.scrape(registry, Map.of("container_ip", "10.0.1.1"));
        assertTrue(text.lines().anyMatch(line -> line.startsWith("calls_total{") && line.contains("role=\"prefill\"") && line.endsWith(" 2.0")));
        for (var line : text.lines().filter(line -> !line.startsWith("#")).toList())
            assertTrue(line.contains("container_ip=\"10.0.1.1\""), line);
        assertTrue(text.contains("le="));
        assertFalse(text.contains("instance="));
        assertFalse(text.contains("engine_port="));
        assertFalse(text.contains("node="));
    }
}
