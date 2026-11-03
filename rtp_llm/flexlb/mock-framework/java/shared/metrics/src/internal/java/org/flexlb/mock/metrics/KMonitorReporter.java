package org.flexlb.mock.metrics;

import com.taobao.kmonitor.*;
import com.taobao.kmonitor.impl.KMonitorConfig;
import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.*;

/** Internal SDK transport, compiled only by the internal-monitoring profile. */
public final class KMonitorReporter implements MetricReporter {
    private final Set<String> registered = new HashSet<>();
    private KMonitor monitor;
    private long interval;

    static long configure(Map<String, String> env) {
        String host = env.getOrDefault("kmonitorSinkAddress", env.getOrDefault("HIPPO_SLAVE_IP", "127.0.0.1"));
        int port = positive(env, "kmonitorPort", 4141);
        if (host.isBlank() || port > 65535) throw new IllegalArgumentException("Invalid KMonitor sink");
        if (host.contains(":") && !host.startsWith("[")) host = "[" + host + "]";
        KMonitorConfig.setSinkAddress(host + ":" + port);
        KMonitorConfig.setKMonitorServiceName(env.getOrDefault("kmonitorServiceName", ""));
        KMonitorConfig.setKMonitorTenantName(env.getOrDefault("kmonitorTenant", "default"));
        KMonitorConfig.setGlobalTags(new ImmutableMetricTags(MetricTags.fromEnvironment(env)));
        KMonitorConfig.setFlumeSinkEnable(true);
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

    @Override public synchronized void initialize(Map<String,String> env) {
        if (monitor != null) throw new IllegalStateException("KMonitor reporter already initialized");
        interval = configure(env);
        try {
            KMonitorFactory.start();
            if (!KMonitorFactory.isStarted()) throw new IllegalStateException("KMonitor SDK failed to start");
            monitor = Objects.requireNonNull(KMonitorFactory.getKMonitor("rtp_llm"), "KMonitor instance unavailable");
        } catch (RuntimeException | Error error) { KMonitorFactory.stop(); throw error; }
        Runtime.getRuntime().addShutdownHook(new Thread(KMonitorFactory::stop, "mock-kmonitor-close"));
        System.getLogger(getClass().getName()).log(System.Logger.Level.INFO, "Engine metrics backend: kmonitor");
    }

    @Override public long intervalMillis() { return interval; }

    @Override public synchronized void report(List<MetricFamilySamples> snapshot) {
        if (monitor == null) throw new IllegalStateException("KMonitor reporter is not initialized");
        for (var family : snapshot) for (var sample : family.samples) {
            if (!Double.isFinite(sample.value)) continue;
            if (registered.add(sample.name)) {
                // These are completed window snapshots, not raw QPS events or Histogram observations.
                if (!monitor.register(sample.name, MetricType.GAUGE)) {
                    registered.remove(sample.name);
                    throw new IllegalStateException("Cannot register KMonitor metric: " + sample.name);
                }
            }
            var tags = new HashMap<String,String>();
            for (int i = 0; i < sample.labelNames.size(); i++) tags.put(sample.labelNames.get(i), sample.labelValues.get(i));
            monitor.report(sample.name, new ImmutableMetricTags(tags), sample.value);
        }
    }
}
