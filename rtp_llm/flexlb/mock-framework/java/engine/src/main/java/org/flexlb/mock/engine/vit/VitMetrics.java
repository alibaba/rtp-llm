package org.flexlb.mock.engine.vit;

import io.prometheus.client.*;
import java.util.*;
import org.flexlb.mock.metrics.*;

/**
 * Per-V instance telemetry; no simulated per-process CPU/GPU utilization.
 */
final class VitMetrics {
    final CollectorRegistry registry = new CollectorRegistry();
    final EventMetrics events;

    VitMetrics() {
        events = new EventMetrics("vit");
        events.register(registry);
        events.observe("py_rtp_vit_qps", 0, Map.of("source", "mm_embedding"));
        for (String name : List.of("py_rtp_vit_success_qps", "py_rtp_vit_error_qps",
                "py_rtp_vit_embedding_cache_hit_qps", "py_rtp_vit_embedding_cache_miss_qps",
                "py_rtp_vit_embedding_cache_inflight_qps", "py_rtp_vit_embedding_cache_eviction_qps",
                "py_rtp_vit_embedding_overload_qps", "py_rtp_vit_embedding_tps")) events.observe(name, 0);
    }

    void started() { events.observe("py_rtp_vit_qps", 1, Map.of("source", "mm_embedding")); }

    void error(Throwable error, VitCache.Claim claim) {
        synchronized(error) {
            boolean reported = Arrays.stream(error.getSuppressed()).anyMatch(ReportedError.class::isInstance);
            boolean entryReported = claim != null && !claim.markErrorReported();
            if (!reported) error.addSuppressed(new ReportedError());
            if (!reported && !entryReported) events.increment("py_rtp_vit_error_qps");
        }
    }

    String scrape(VitScheduler scheduler, VitCache cache, VitPreparation preparation) {
        var text = new java.io.StringWriter();
        try { io.prometheus.client.exporter.common.TextFormat.write004(text,
            Collections.enumeration(collect(scheduler, cache, preparation))); }
        catch (java.io.IOException impossible) { throw new java.io.UncheckedIOException(impossible); }
        return text.toString();
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

    // Like real's entry/exception markers: shared work and its RPC waiters report one failure.
    /**
     * Marks a ViT failure whose error metric has already been reported.
     */
    private static final class ReportedError extends Throwable {
        ReportedError() { super(null, null, false, false); }
    }
}
