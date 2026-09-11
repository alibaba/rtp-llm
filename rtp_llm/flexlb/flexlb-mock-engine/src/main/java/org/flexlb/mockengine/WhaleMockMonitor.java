package org.flexlb.mockengine;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MasterStatusProvider;
import org.flexlb.metric.NoOpFlexMonitor;

import java.util.Map;
import java.util.HashMap;

/** Optional internal adapter; no private class is linked by the open-source runtime. */
final class WhaleMockMonitor implements AutoCloseable {
    private final FlexMonitor monitor;
    private final Map<String, Long> previous = new HashMap<>();
    private final java.util.Set<String> registered = new java.util.HashSet<>();
    private long sampledAt = System.nanoTime();

    static WhaleMockMonitor create() {
        try {
            FlexMonitor monitor = (FlexMonitor) Class.forName("org.flexlb.monitor.FlexMonitorFactory")
                    .getMethod("createKMonitorAdapter", MasterStatusProvider.class).invoke(null, new Object[]{null});
            if (monitor instanceof NoOpFlexMonitor) {
                throw new IllegalStateException("KMonitor initialization returned a no-op adapter");
            }
            return new WhaleMockMonitor(monitor);
        } catch (ReflectiveOperationException | LinkageError error) {
            throw new IllegalStateException("Whale KMonitor requires the internal Maven profile", error);
        }
    }

    WhaleMockMonitor(FlexMonitor monitor) { this.monitor = monitor; }

    void sample(JavaMockEngineCluster.FastRpcService service) {
        long now = System.nanoTime();
        double seconds = Math.max(1e-9, (now - sampledAt) / 1e9);
        sampledAt = now;
        FlexMetricTags tags = new FlexMetricTags.ImmutableFlexMetricTags(service.whaleMetricTags());
        service.whaleMetrics().forEach((name, value) -> {
            if (registered.add(name)) monitor.register(name, FlexMetricType.GAUGE);
            monitor.report(name, tags, value.doubleValue());
            String rate = switch (name) {
                case "mock_context_compute_tokens_total" -> "rtp_llm_context_tps";
                case "mock_context_tokens_total" -> "rtp_llm_context_tps_with_cache";
                case "mock_generate_tokens_total" -> "rtp_llm_generate_tps";
                default -> null;
            };
            if (rate != null) {
                Long before = previous.put(name, value.longValue());
                if (registered.add(rate)) monitor.register(rate, FlexMetricType.GAUGE);
                monitor.report(rate, tags, Math.max(0, value.longValue() - (before == null ? 0 : before)) / seconds);
            }
        });
    }

    @Override
    public void close() { monitor.close(); }
}
