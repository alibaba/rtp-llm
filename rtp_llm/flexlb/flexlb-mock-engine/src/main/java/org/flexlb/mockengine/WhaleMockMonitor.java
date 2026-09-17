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
    private static final class EngineSample {
        final Map<String, Long> previous = new HashMap<>();
        boolean schedulerReported;
        long sampledAt = System.nanoTime();
    }
    private final Map<Map<String, String>, EngineSample> samples = new HashMap<>();

    private EngineSample state(Map<String, String> labels) {
        return samples.computeIfAbsent(Map.copyOf(labels), ignored -> new EngineSample());
    }
    private final java.util.Set<String> registered = new java.util.HashSet<>();
    private static final java.util.Set<String> STEP_METRICS = java.util.Set.of(
            "rtp_llm_running_stream_size", "rtp_llm_context_batch_size", "rtp_llm_generate_batch_size");

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
        sample(service.whaleMetrics(), service.whaleMetricTags(), System.nanoTime(), service.autoFetchEnabled());
    }

    synchronized void reportEvent(Map<String, Number> metrics, Map<String, String> labels) {
        FlexMetricTags tags = new FlexMetricTags.ImmutableFlexMetricTags(labels);
        metrics.forEach((name, value) -> {
            if (registered.add(name)) monitor.register(name, FlexMetricType.GAUGE);
            monitor.report(name, tags, value.doubleValue());
            if (name.equals("rtp_llm_first_token_latency_us")
                    && "ROLE_TYPE_PREFILL".equals(labels.get("role"))) {
                String alias = "py_rtp_response_first_token_rt";
                if (registered.add(alias)) monitor.register(alias, FlexMetricType.GAUGE);
                monitor.report(alias, dashboardTags(labels), value.doubleValue() / 1000.0);
            }
        });
    }

    // Keep the platform's normal selection tags. The P/D distinction on these
    // dashboard aliases comes from hippo_role alone, not mock-only source or
    // logical-engine labels.
    private static FlexMetricTags dashboardTags(Map<String, String> labels) {
        Map<String, String> dashboard = new HashMap<>();
        for (String key : java.util.List.of(
                "hippo_app", "hippo_group", "hippo_role", "host_ip", "container_ip", "dp_rank")) {
            String value = labels.get(key);
            if (value != null) dashboard.put(key, value);
        }
        return new FlexMetricTags.ImmutableFlexMetricTags(dashboard);
    }

    synchronized void reportScheduler(Map<String, Number> metrics, Map<String, String> labels) {
        FlexMetricTags tags = new FlexMetricTags.ImmutableFlexMetricTags(labels);
        for (var entry : metrics.entrySet()) {
            if (!STEP_METRICS.contains(entry.getKey()))
                throw new IllegalArgumentException("Not a scheduler metric: " + entry.getKey());
            if (registered.add(entry.getKey())) monitor.register(entry.getKey(), FlexMetricType.GAUGE);
            monitor.report(entry.getKey(), tags, entry.getValue().doubleValue());
        }
        state(labels).schedulerReported = true;
    }

    synchronized void sample(Map<String, Number> metrics, Map<String, String> labels, long now) {
        sample(metrics, labels, now, false);
    }

    synchronized void sample(Map<String, Number> metrics, Map<String, String> labels,
                             long now, boolean noFetch) {
        EngineSample sample = state(labels);
        double seconds = Math.max(1e-9, (now - sample.sampledAt) / 1e9);
        sample.sampledAt = now;
        FlexMetricTags tags = new FlexMetricTags.ImmutableFlexMetricTags(labels);
        boolean hadSchedulerSteps = sample.schedulerReported;
        sample.schedulerReported = false;
        Map<String, Long> deltas = new HashMap<>();
        metrics.forEach((name, value) -> {
            if (name.endsWith("_total")) {
                Long before = sample.previous.put(name, value.longValue());
                deltas.put(name, before == null ? 0L : Math.max(0, value.longValue() - before));
            }
        });
        // No-Fetch mode has no client success response. This is the engine's
        // successful Decode terminal rate, separate from frontend success QPS.
        if ("ROLE_TYPE_DECODE".equals(labels.get("role"))
                && metrics.containsKey("mock_completed_requests_total")) {
            String name = "mock_decode_success_qps";
            if (registered.add(name)) monitor.register(name, FlexMetricType.GAUGE);
            double successQps = deltas.getOrDefault("mock_completed_requests_total", 0L) / seconds;
            monitor.report(name, tags, successQps);
            if (noFetch) {
                // Dashboard-compatible alias for successful Decode terminals.
                // hippo_role separates it from the frontend's response rate.
                String alias = "py_rtp_success_qps_metric";
                if (registered.add(alias)) monitor.register(alias, FlexMetricType.QPS);
                monitor.report(alias, dashboardTags(labels),
                        deltas.getOrDefault("mock_completed_requests_total", 0L));
            }
        }
        metrics.forEach((name, value) -> {
            // Preserve execution-round samples. A periodic zero between two
            // short P batches must not dilute them. With no rounds this period,
            // retain the instantaneous gauge so idle engines return to zero.
            if (name.equals("rtp_llm_context_batch_size")) return;
            // P running size is sampled at execution transitions. FIFO blocks
            // while idle; periodic zeros would bias low-traffic averages.
            // Completion already emits the terminal zero, so preserve that event.
            if (name.equals("rtp_llm_running_stream_size")
                    && "ROLE_TYPE_PREFILL".equals(labels.get("role"))) return;
            if (hadSchedulerSteps && STEP_METRICS.contains(name)) return;
            if (registered.add(name)) monitor.register(name, FlexMetricType.GAUGE);
            monitor.report(name, tags, value.doubleValue());
            String rate = switch (name) {
                case "mock_context_compute_tokens_total" -> "rtp_llm_context_tps";
                case "mock_context_tokens_total" -> "rtp_llm_context_tps_with_cache";
                case "mock_decode_step_tokens_total" -> "rtp_llm_generate_tps";
                default -> null;
            };
            if (rate != null) {
                if (registered.add(rate)) monitor.register(rate, FlexMetricType.GAUGE);
                double wallTps = deltas.getOrDefault(name, 0L) / seconds;
                String duration = switch (name) {
                    case "mock_context_compute_tokens_total" -> "mock_context_compute_ms_total";
                    case "mock_context_tokens_total" -> "mock_context_with_cache_ms_total";
                    default -> null;
                };
                long executionMs = duration == null ? 0 : deltas.getOrDefault(duration, 0L);
                double tps = duration == null ? wallTps : executionMs > 0
                        ? deltas.getOrDefault(name, 0L) * 1000.0 / executionMs : 0;
                monitor.report(rate, tags, tps);
                String wall = switch (name) {
                    case "mock_context_compute_tokens_total" -> "rtp_llm_context_wall_tps";
                    case "mock_context_tokens_total" -> "rtp_llm_context_wall_tps_with_cache";
                    default -> null;
                };
                if (wall != null) {
                    if (registered.add(wall)) monitor.register(wall, FlexMetricType.GAUGE);
                    monitor.report(wall, tags, wallTps);
                }
            }
        });
    }

    static Map<String, String> engineTags(Map<String, String> environment, String host, String role) {
        Map<String, String> tags = new HashMap<>();
        tags.put("hippo_app", environment.getOrDefault("HIPPO_APP", ""));
        tags.put("hippo_role", environment.getOrDefault("HIPPO_ROLE", ""));
        tags.put("hippo_group", environment.getOrDefault("HIPPO_SERVICE_NAME", ""));
        tags.put("host_ip", environment.getOrDefault("HIPPO_SLAVE_IP", host));
        tags.put("container_ip", host);
        tags.put("dp_rank", "0"); // Each mock engine is single-DP; engine identity is a separate tag.
        tags.put("priority", "0"); // Aggregate mock series, not a per-priority breakdown.
        tags.put("mtp_model_type", "main");
        tags.put("pool", "0"); // The mock has one physical KV pool, matching C++ gid=0.
        String aliasVariable = switch (role) {
            case "ROLE_TYPE_PREFILL" -> "MOCK_PREFILL_HIPPO_ROLE";
            case "ROLE_TYPE_DECODE" -> "MOCK_DECODE_HIPPO_ROLE";
            default -> null;
        };
        String alias = aliasVariable == null ? null : environment.get(aliasVariable);
        if (alias != null && !alias.isBlank()) {
            tags.put("hippo_role", alias);
        }
        return tags;
    }

    @Override
    public void close() { monitor.close(); }
}
