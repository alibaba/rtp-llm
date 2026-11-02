package org.flexlb.mock.metrics;

import io.prometheus.client.CollectorRegistry;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WindowMetricsTest {
    private static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }

    @Test void windowsUseEventMeanAndWallQpsAndDoNotRepeatIdleLatency() {
        var clock = new AtomicLong(); var registry = new CollectorRegistry();
        var metrics = new WindowMetrics("vit", clock::get); metrics.register(registry);
        metrics.observe("py_rtp_vit_embedding_rt", 10); metrics.observe("py_rtp_vit_embedding_rt", 30);
        metrics.increment("py_rtp_vit_qps"); metrics.increment("py_rtp_vit_qps");
        clock.set(2_000_000_000L);
        var text = MetricTags.scrape(registry, Map.of());
        assertEquals(20, sample(text, "py_rtp_vit_embedding_rt", ""));
        assertEquals(1, sample(text, "py_rtp_vit_qps", ""));
        assertEquals(40, sample(text, "py_rtp_vit_embedding_rt_sum", ""));
        clock.set(3_000_000_000L); text = MetricTags.scrape(registry, Map.of());
        assertFalse(text.contains("py_rtp_vit_embedding_rt{"));
        assertEquals(0, sample(text, "py_rtp_vit_qps", ""));
        assertEquals(2, sample(text, "py_rtp_vit_embedding_rt_count", ""));
        assertThrows(IllegalArgumentException.class, () -> metrics.observe("x", Double.NaN));
    }
}
