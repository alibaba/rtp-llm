package org.flexlb.metric;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import javax.management.Notification;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import java.lang.management.GarbageCollectorMXBean;
import java.util.List;

import static org.flexlb.constant.MetricConstant.JVM_GC_COLLECTION_COUNT;
import static org.flexlb.constant.MetricConstant.JVM_GC_PAUSE_TOTAL_MS;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.isNull;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

class JvmGcMetricsReporterTest {

    private final SimpleMeterRegistry registry = new SimpleMeterRegistry();

    @AfterEach
    void closeRegistry() {
        registry.close();
    }

    @Test
    void recordsWindowCountsAndEventWeightedMeanWithoutIdleSamples() {
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        JvmGcMetricsReporter reporter = new JvmGcMetricsReporter(monitor,
                List.of(collector("G1 Young Generation"), collector("G1 Old Generation"),
                        collector("G1 Concurrent GC")));
        reporter.init();
        try {
            double countBefore = counter(registry, JVM_GC_COLLECTION_COUNT, "young").count();
            double timeBefore = counter(registry, JVM_GC_PAUSE_TOTAL_MS, "young").count();
            reporter.reportPause("G1 Young Generation", 10L);
            reporter.reportHeartbeat();
            reporter.reportPause("G1 Young Generation", 30L);
            reporter.reportHeartbeat();
            reporter.reportPause("G1 Old Generation", 100L);

            double windowCount = counter(registry, JVM_GC_COLLECTION_COUNT, "young").count() - countBefore;
            double windowTime = counter(registry, JVM_GC_PAUSE_TOTAL_MS, "young").count() - timeBefore;
            assertEquals(2.0D, windowCount);
            assertEquals(20.0D, windowTime / windowCount);
            assertEquals(1.0D, counter(registry, JVM_GC_COLLECTION_COUNT, "full").count());
            assertEquals(100.0D, counter(registry, JVM_GC_PAUSE_TOTAL_MS, "full").count());
            assertEquals(0.0D, counter(registry, JVM_GC_COLLECTION_COUNT, "concurrent").count());
        } finally {
            reporter.close();
        }
    }

    @Test
    void retainsZeroDurationCollectionsAndIgnoresInvalidEvents() {
        JvmGcMetricsReporter reporter = new JvmGcMetricsReporter(new MicrometerFlexMonitor(registry),
                List.of(collector("G1 Young Generation")));
        reporter.init();
        try {
            reporter.reportPause("G1 Young Generation", 0L);
            reporter.reportPause("G1 Young Generation", -1L);
            reporter.reportPause("not a registered collector", 100L);
            assertEquals(1.0D, counter(registry, JVM_GC_COLLECTION_COUNT, "young").count());
            assertEquals(0.0D, counter(registry, JVM_GC_PAUSE_TOTAL_MS, "young").count());
        } finally {
            reporter.close();
        }
    }

    @Test
    void removesGcNotificationListenerWhenReporterCloses() throws Exception {
        GarbageCollectorMXBean bean = collector("G1 Young Generation");
        FlexMonitor monitor = mock(FlexMonitor.class);
        JvmGcMetricsReporter reporter = new JvmGcMetricsReporter(monitor, List.of(bean));
        reporter.init();
        ArgumentCaptor<NotificationListener> listener = ArgumentCaptor.forClass(NotificationListener.class);
        verify((NotificationEmitter) bean).addNotificationListener(listener.capture(), isNull(), isNull());
        listener.getValue().handleNotification(new Notification("unrelated", "test", 1L), null);
        reporter.close();
        verify((NotificationEmitter) bean).removeNotificationListener(listener.getValue());
    }

    private static GarbageCollectorMXBean collector(String name) {
        GarbageCollectorMXBean bean = mock(GarbageCollectorMXBean.class,
                withSettings().extraInterfaces(NotificationEmitter.class));
        when(bean.getName()).thenReturn(name);
        return bean;
    }

    private static Counter counter(SimpleMeterRegistry registry, String metric, String type) {
        return registry.get("flexlb." + metric).tag("gc", type).counter();
    }
}
