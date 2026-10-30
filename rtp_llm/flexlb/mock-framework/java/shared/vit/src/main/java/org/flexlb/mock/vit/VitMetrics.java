package org.flexlb.mock.vit;

import io.prometheus.client.*;
import java.util.*;
import org.flexlb.mock.metrics.*;

/** Per-V instance telemetry; no simulated per-process CPU/GPU utilization. */
final class VitMetrics {
    final CollectorRegistry registry = new CollectorRegistry();
    final WindowMetrics events = new WindowMetrics("vit");
    VitMetrics() {
        events.register(registry);
        for (String name : List.of("py_rtp_vit_qps", "py_rtp_vit_success_qps", "py_rtp_vit_error_qps",
                "py_rtp_vit_embedding_cache_hit_qps", "py_rtp_vit_embedding_cache_miss_qps",
                "py_rtp_vit_embedding_cache_inflight_qps", "py_rtp_vit_embedding_cache_eviction_qps",
                "py_rtp_vit_embedding_overload_qps")) events.observe(name, 0);
    }
    String scrape(VitScheduler scheduler, VitCache cache) {
        var state = new TreeMap<>(scheduler.snapshot()); state.putAll(cache.snapshot());
        var gauges = new Collector() {
            @Override public List<MetricFamilySamples> collect() {
                var result = new ArrayList<MetricFamilySamples>();
                state.forEach((name, value) -> {
                    var gauge = new GaugeMetricFamily(name, "Current V instance state.", List.of("role"));
                    gauge.addMetric(List.of("vit"), value); result.add(gauge);
                });
                return result;
            }
        };
        var snapshot = new CollectorRegistry(); gauges.register(snapshot);
        var tags = MetricTags.fromEnvironment(System.getenv());
        return MetricTags.scrape(registry, tags) + MetricTags.scrape(snapshot, tags);
    }
}
