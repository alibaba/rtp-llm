package org.flexlb.metric;

import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class MicrometerFlexMonitorTest {
    @Test
    void lazyMeasurementsAreNotComputedForDisabledMetrics() {
        var evaluations = new java.util.concurrent.atomic.AtomicInteger();
        var registry = new SimpleMeterRegistry();
        var monitor = new MicrometerFlexMonitor(registry);
        try {
            MicrometerFlexMonitor.setAllowedMetrics(java.util.Set.of("enabled"));
            java.util.function.DoubleSupplier measurement = () -> evaluations.incrementAndGet();
            monitor.reportLazy("disabled", null, measurement);
            NoOpFlexMonitor.getInstance().reportLazy("enabled", null, measurement);
            assertEquals(0, evaluations.get());
            monitor.reportLazy("enabled", null, measurement);
            assertEquals(1, evaluations.get());
            assertEquals(1, registry.find("flexlb.enabled").gauge().value());
        } finally { monitor.close(); registry.close(); }
    }

    @AfterEach
    void resetAllowedMetrics() {
        MicrometerFlexMonitor.setAllowedMetrics(null);
    }

    @Test
    void prepareTimerRegistersWithoutRecordingAndReportReusesIt() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            FlexMetricTags tags = FlexMetricTags.of("role", "PREFILL", "engineIp", "127.0.0.1");
            monitor.register("test.timer", FlexMetricType.TIMER);

            monitor.prepare("test.timer", tags);

            Timer prepared = registry.find("flexlb.test.timer")
                    .tags("role", "PREFILL", "engineIp", "127.0.0.1")
                    .timer();
            assertNotNull(prepared);
            assertEquals(0L, prepared.count());

            monitor.report("test.timer", tags, 12.0D);

            Timer reported = registry.find("flexlb.test.timer")
                    .tags("role", "PREFILL", "engineIp", "127.0.0.1")
                    .timer();
            assertEquals(prepared, reported);
            assertEquals(1L, reported.count());
            assertEquals(12.0D, reported.totalTime(TimeUnit.MILLISECONDS), 0.001D);
        } finally {
            monitor.close();
            registry.close();
        }
    }
}
