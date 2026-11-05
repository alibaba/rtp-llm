package org.flexlb.mock.metrics;

import static org.junit.jupiter.api.Assertions.*;
import com.taobao.kmonitor.impl.KMonitorConfig;
import com.taobao.kmonitor.*;
import com.taobao.kmonitor.metric.GaugeMetric;
import com.taobao.kmonitor.core.MetricsInfo;
import com.taobao.kmonitor.core.MetricsRecordBuilder;
import io.prometheus.client.GaugeMetricFamily;
import io.prometheus.client.CounterMetricFamily;
import java.lang.reflect.Proxy;
import java.util.*;
import org.junit.jupiter.api.Test;

class KMonitorReporterTest {
    private static final class NativeGauge {
        final GaugeMetric gauge = new GaugeMetric("input_length", 0);
        final Map<String, MetricType> types = new HashMap<>();
        final Map<String, List<Double>> values = new HashMap<>();
        final KMonitorReporter reporter;
        NativeGauge() {
            var monitor = (KMonitor) Proxy.newProxyInstance(KMonitor.class.getClassLoader(), new Class<?>[]{KMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("register")) { types.put((String) args[0], (MetricType) args[1]); return true; }
                    if (method.getName().equals("report")) {
                        var name = (String) args[0]; double value = (double) args[2];
                        values.computeIfAbsent(name, ignored -> new ArrayList<>()).add(value);
                        if (name.equals("input_length")) gauge.update(value);
                        return null;
                    }
                    throw new AssertionError(method);
                });
            reporter = new KMonitorReporter(monitor);
        }
        double[] payload() {
            var builder = new MetricsRecordBuilder(new MetricsInfo("test"));
            gauge.snapshot(builder, true);
            var value = builder.getRecord().metrics().getFirst().value().getValue().toString();
            return Arrays.stream(value.split("_")).mapToDouble(Double::parseDouble).toArray();
        }
    }

    @Test void nativeSdkSerializationRetainsUnequalInstanceSampleWeights() {
        var first = new NativeGauge(); var second = new NativeGauge();
        var a = new EventMetrics("prefill", System::nanoTime, first.reporter, Map.of("dp_rank", "0"));
        var b = new EventMetrics("prefill", System::nanoTime, second.reporter, Map.of("dp_rank", "0"));
        for (int i = 0; i < 100; i++) a.observe("input_length", 10);
        b.observe("input_length", 1000);
        first.reporter.report(a.collect()); second.reporter.report(b.collect());
        assertEquals(100, first.values.get("input_length").size());
        assertEquals(1, second.values.get("input_length").size());
        // Native Gauge payload is average_sum_max_min_count, consumed by global_avg.
        var x = first.payload(); var y = second.payload();
        assertArrayEquals(new double[]{10, 1000, 10, 10, 100}, x);
        assertArrayEquals(new double[]{1000, 1000, 1000, 1000, 1}, y);
        assertEquals(2000.0 / 101, (x[1] + y[1]) / (x[4] + y[4]));
        assertEquals(MetricType.GAUGE, first.types.get("input_length"));
        assertFalse(first.types.containsKey("input_length_sum"));
    }

    @Test void qpsUsesRawCountsAndPeriodicSnapshotsRemainGauges() {
        var sdk = new NativeGauge();
        var clock = new java.util.concurrent.atomic.AtomicLong();
        var events = new EventMetrics("vit", clock::get, sdk.reporter, Map.of());
        events.increment("vit_qps"); events.increment("vit_qps");
        clock.set(2_000_000_000L);
        sdk.reporter.report(events.collect());
        assertEquals(List.of(1.0, 1.0), sdk.values.get("vit_qps"));
        assertEquals(MetricType.QPS, sdk.types.get("vit_qps"));
        var state = new GaugeMetricFamily("context_tps", "Rate already computed by the engine.", List.of("role"));
        state.addMetric(List.of("prefill"), 123);
        var helper = new CounterMetricFamily("local_helper_total", "Prometheus cumulative helper.", List.of("role"));
        helper.addMetric(List.of("prefill"), 999);
        sdk.reporter.report(List.of(state, helper));
        assertEquals(MetricType.GAUGE, sdk.types.get("context_tps"));
        assertEquals(List.of(123.0), sdk.values.get("context_tps"));
        assertFalse(sdk.types.containsKey("local_helper_total"));
        assertFalse(sdk.values.containsKey("local_helper_total"));
    }

    @Test void usesNativeWhaleConfigurationWithoutFlexlbOverrides() {
        long interval = KMonitorReporter.configure(Map.of("kmonitorServiceName", "engine-service",
            "kmonitorTenant", "tenant", "kmonitorSinkAddress", "10.1.2.3", "kmonitorPort", "4142",
            "kmonitorNormalSamplePeriod", "2"));
        assertEquals("engine-service", KMonitorConfig.getKMonitorServiceName());
        assertEquals("tenant", KMonitorConfig.getKMonitorTenantName());
        assertEquals("10.1.2.3:4142", KMonitorConfig.getSinkAddress());
        assertEquals(2000, interval);
        assertEquals(1000, KMonitorReporter.configure(Map.of("HIPPO_SLAVE_IP", "10.2.3.4")));
        assertEquals("10.2.3.4:4141", KMonitorConfig.getSinkAddress());
        assertThrows(IllegalArgumentException.class, () -> KMonitorReporter.configure(Map.of("kmonitorPort", "0")));
    }
}
