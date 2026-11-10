package org.flexlb.mock.metrics;

import com.taobao.kmonitor.*;
import com.taobao.kmonitor.impl.KMonitorConfig;
import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.*;

/**
 * Maps event kinds and state snapshots to native KMonitor metrics.
 */
public final class KMonitorReporter implements MetricReporter {
    private final Map<String, MetricType> registered = new HashMap<>();

    private KMonitor monitor;

    private long interval;

    public KMonitorReporter() { }

    KMonitorReporter(KMonitor monitor) { this.monitor = Objects.requireNonNull(monitor); }

    @Override public synchronized void initialize(Map<String,String> env) {
        if (monitor != null) throw new IllegalStateException("KMonitor reporter already initialized");
        interval = configure(env);
        try {
            KMonitorFactory.start();
            if (!KMonitorFactory.isStarted()) throw new IllegalStateException("KMonitor SDK failed to start");
            // Match native FlexMonitorFactory: install global tags after start, before creating the monitor.
            KMonitorFactory.addGlobalTags(new ImmutableMetricTags(MetricTags.fromEnvironment(env)));
            monitor = Objects.requireNonNull(KMonitorFactory.getKMonitor("rtp_llm"), "KMonitor instance unavailable");
        } catch (RuntimeException | Error error) { KMonitorFactory.stop(); throw error; }
        Runtime.getRuntime().addShutdownHook(new Thread(KMonitorFactory::stop, "mock-kmonitor-close"));
        System.getLogger(getClass().getName()).log(System.Logger.Level.INFO, "Engine metrics backend: kmonitor");
    }

    @Override public long intervalMillis() { return interval; }

    @Override public void observe(String name, EventKind kind, Map<String, String> tags, double value) {
        if (monitor == null) throw new IllegalStateException("KMonitor reporter is not initialized");
        register(name, kind == EventKind.DELTA ? MetricType.QPS : MetricType.GAUGE);
        monitor.report(name, new ImmutableMetricTags(tags), value);
    }

    @Override public void report(List<MetricFamilySamples> snapshot) {
        if (monitor == null) throw new IllegalStateException("KMonitor reporter is not initialized");
        for (var family : snapshot) {
            // Prometheus counters/histograms are local query helpers, not native Gauge observations.
            if (family.type != io.prometheus.client.Collector.Type.GAUGE) continue;
            for (var sample : family.samples) {
                if (!Double.isFinite(sample.value)) continue;
                register(sample.name, MetricType.GAUGE);
                var tags = new HashMap<String,String>();
                for (int i = 0; i < sample.labelNames.size(); i++) tags.put(sample.labelNames.get(i), sample.labelValues.get(i));
                monitor.report(sample.name, new ImmutableMetricTags(tags), sample.value);
            }
        }
    }

    static long configure(Map<String, String> env) {
        String host = env.getOrDefault("kmonitorSinkAddress", env.getOrDefault("HIPPO_SLAVE_IP", "127.0.0.1"));
        int port = positive(env, "kmonitorPort", 4141);
        if (host.isBlank() || port > 65535) throw new IllegalArgumentException("Invalid KMonitor sink");
        if (host.contains(":") && !host.startsWith("[")) host = "[" + host + "]";
        KMonitorConfig.setSinkAddress(host + ":" + port);
        KMonitorConfig.setKMonitorServiceName(env.getOrDefault("kmonitorServiceName", ""));
        KMonitorConfig.setKMonitorTenantName(env.getOrDefault("kmonitorTenant", "default"));
        MetricTags.fromEnvironment(env); // Validate native tags before starting the SDK.
        KMonitorConfig.setPrometheusExport(false);
        // SDK built-ins describe this JVM, not each simulated Engine.
        KMonitorConfig.setSystemMetricsEnabled(false);
        return Math.multiplyExact((long) positive(env, "kmonitorNormalSamplePeriod", 1), 1000);
    }

    private static int positive(Map<String,String> env, String key, int fallback) {
        try {
            int value = Integer.parseInt(env.getOrDefault(key, Integer.toString(fallback)));
            if (value > 0) return value;
        } catch (NumberFormatException ignored) { }
        throw new IllegalArgumentException("Invalid " + key);
    }

    private void register(String name, MetricType type) {
        var previous = registered.get(name);
        if (previous != null) {
            if (previous != type) throw new IllegalStateException("Inconsistent KMonitor metric type: " + name);
            return;
        }
        if (!monitor.register(name, type)) throw new IllegalStateException("Cannot register KMonitor metric: " + name);
        registered.put(name, type);
    }
}
