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
            // Reuse internal sink initialization, but not the master's whale-lb
            // namespace. Engine dashboards query unprefixed rtp_llm_* metrics.
            Class<?> config = Class.forName("com.taobao.kmonitor.impl.KMonitorConfig");
            config.getMethod("setKMonitorServiceName", String.class).invoke(null, "");
            Object engineMonitor = Class.forName("com.taobao.kmonitor.KMonitorFactory")
                    .getMethod("getKMonitor", String.class, String.class, String.class)
                    .invoke(null, "rtp_llm_mock", "", System.getenv().getOrDefault("kmonitorTenant", "default"));
            FlexMonitor adapter = (FlexMonitor) Class.forName("org.flexlb.monitor.KMonitorAdapter")
                    .getConstructor(Class.forName("com.taobao.kmonitor.KMonitor"))
                    .newInstance(engineMonitor);
            return new WhaleMockMonitor(adapter);
        } catch (ReflectiveOperationException | LinkageError error) {
            throw new IllegalStateException("Whale KMonitor requires the internal Maven profile", error);
        }
    }

    WhaleMockMonitor(FlexMonitor monitor) { this.monitor = monitor; }

    void sample(JavaMockEngineCluster.FastRpcService service) {
        sample(service.whaleMetrics(), service.whaleMetricTags(), System.nanoTime());
    }

    void sample(Map<String, Number> metrics, Map<String, String> labels, long now) {
        double seconds = Math.max(1e-9, (now - sampledAt) / 1e9);
        sampledAt = now;
        FlexMetricTags tags = new FlexMetricTags.ImmutableFlexMetricTags(labels);
        metrics.forEach((name, value) -> {
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
                double tps = Math.max(0, value.longValue() - (before == null ? 0 : before)) / seconds;
                monitor.report(rate, tags, tps);
                String wall = switch (name) {
                    case "mock_context_compute_tokens_total" -> "rtp_llm_context_wall_tps";
                    case "mock_context_tokens_total" -> "rtp_llm_context_wall_tps_with_cache";
                    default -> null;
                };
                if (wall != null) {
                    if (registered.add(wall)) monitor.register(wall, FlexMetricType.GAUGE);
                    monitor.report(wall, tags, tps);
                }
            }
        });
    }

    static Map<String, String> engineTags(Map<String, String> environment, String host) {
        Map<String, String> tags = new HashMap<>();
        tags.put("hippo_app", environment.getOrDefault("HIPPO_APP", ""));
        tags.put("hippo_role", environment.getOrDefault("HIPPO_ROLE", ""));
        tags.put("hippo_group", environment.getOrDefault("HIPPO_SERVICE_NAME", ""));
        tags.put("host_ip", environment.getOrDefault("HIPPO_SLAVE_IP", host));
        tags.put("container_ip", host);
        tags.put("dp_rank", "0"); // Whale mode enforces one engine per Pod.
        tags.put("priority", "0"); // Aggregate mock series, not a per-priority breakdown.
        tags.put("mtp_model_type", "main");
        return tags;
    }

    @Override
    public void close() { monitor.close(); }
}
