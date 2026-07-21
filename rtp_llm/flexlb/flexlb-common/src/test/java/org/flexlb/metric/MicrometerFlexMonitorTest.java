package org.flexlb.metric;

import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class MicrometerFlexMonitorTest {

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

    @Test
    void timerDoesNotPublishHistogramBuckets() {
        // Histogram buckets are hardcoded to false; only quantile gauges should be published.
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        MicrometerFlexMonitor monitor = new MicrometerFlexMonitor(registry);
        try {
            monitor.register("test.timer", FlexMetricType.TIMER);
            monitor.report("test.timer", null, 5.0);

            // Timer is registered and records the value
            Timer timer = registry.find("flexlb.test.timer").timer();
            assertNotNull(timer, "Timer should be registered");
            assertEquals(1L, timer.count());

            // Quantile gauges should still be present
            assertNotNull(registry.find("flexlb.test.timer.percentile").tag("phi", "0.5").gauge(),
                    "P50 percentile gauge should exist");
            assertNotNull(registry.find("flexlb.test.timer.percentile").tag("phi", "0.99").gauge(),
                    "P99 percentile gauge should exist");

            // Histogram bucket gauges should NOT be present
            assertNull(registry.find("flexlb.test.timer.histogram").gauge(),
                    "Histogram bucket gauges should not exist");
        } finally {
            monitor.close();
            registry.close();
        }
    }
}
