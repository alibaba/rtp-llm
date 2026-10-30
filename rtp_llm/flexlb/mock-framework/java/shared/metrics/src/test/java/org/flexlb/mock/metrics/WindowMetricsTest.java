package org.flexlb.mock.metrics;

import io.prometheus.client.CollectorRegistry;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class WindowMetricsTest {
    @Test void windowsUseEventMeanAndWallQpsAndDoNotRepeatIdleLatency() {
        var clock = new AtomicLong(); var registry = new CollectorRegistry();
        var metrics = new WindowMetrics("vit", clock::get); metrics.register(registry);
        metrics.observe("py_rtp_vit_embedding_rt", 10); metrics.observe("py_rtp_vit_embedding_rt", 30);
        metrics.increment("py_rtp_vit_qps"); metrics.increment("py_rtp_vit_qps");
        clock.set(2_000_000_000L);
        var text = MetricTags.scrape(registry, Map.of());
        assertEquals(20, EngineMetricsTest.sample(text, "py_rtp_vit_embedding_rt", ""));
        assertEquals(1, EngineMetricsTest.sample(text, "py_rtp_vit_qps", ""));
        assertEquals(40, EngineMetricsTest.sample(text, "py_rtp_vit_embedding_rt_sum", ""));
        clock.set(3_000_000_000L); text = MetricTags.scrape(registry, Map.of());
        assertFalse(text.contains("py_rtp_vit_embedding_rt{"));
        assertEquals(0, EngineMetricsTest.sample(text, "py_rtp_vit_qps", ""));
        assertEquals(2, EngineMetricsTest.sample(text, "py_rtp_vit_embedding_rt_count", ""));
        assertThrows(IllegalArgumentException.class, () -> metrics.observe("x", Double.NaN));
    }
}
