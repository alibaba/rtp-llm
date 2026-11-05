package org.flexlb.mock.engine.vit;

import io.prometheus.client.*;
import java.util.*;
import org.flexlb.mock.metrics.*;

/** Per-V instance telemetry; no simulated per-process CPU/GPU utilization. */
final class VitMetrics {
    final CollectorRegistry registry = new CollectorRegistry();
    final EventMetrics events = new EventMetrics("vit");
    VitMetrics() {
        events.register(registry);
        for (String name : List.of("py_rtp_vit_qps", "py_rtp_vit_success_qps", "py_rtp_vit_error_qps",
                "py_rtp_vit_embedding_cache_hit_qps", "py_rtp_vit_embedding_cache_miss_qps",
                "py_rtp_vit_embedding_cache_inflight_qps", "py_rtp_vit_embedding_cache_eviction_qps",
                "py_rtp_vit_embedding_overload_qps")) events.observe(name, 0);
    }
    List<Collector.MetricFamilySamples> collect(VitScheduler scheduler, VitCache cache, VitPreparation preparation) {
        var state = new TreeMap<>(scheduler.snapshot()); state.putAll(cache.snapshot());
        state.put("py_rtp_vit_preprocess_queue_size",(double)preparation.preparing());
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
        var result = new ArrayList<>(MetricTags.collect(registry, tags));
        result.addAll(MetricTags.collect(snapshot, tags));
        return List.copyOf(result);
    }
    String scrape(VitScheduler scheduler, VitCache cache, VitPreparation preparation) {
        var text = new java.io.StringWriter();
        try { io.prometheus.client.exporter.common.TextFormat.write004(text,
            Collections.enumeration(collect(scheduler, cache, preparation))); }
        catch (java.io.IOException impossible) { throw new java.io.UncheckedIOException(impossible); }
        return text.toString();
    }
}
