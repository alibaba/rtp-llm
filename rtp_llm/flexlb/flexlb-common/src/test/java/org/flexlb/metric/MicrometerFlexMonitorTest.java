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

    @AfterEach
    void resetAllowedMetrics() {
        MicrometerFlexMonitor.setAllowedMetrics(null);
    }

    @Test
    void boundBytesDistributionAndFractionalTimerRetainTheirUnits() {
        SimpleMeterRegistry registry=new SimpleMeterRegistry();
        try {
            MicrometerFlexMonitor monitor=new MicrometerFlexMonitor(registry);
            monitor.register("test.bytes",FlexMetricType.DISTRIBUTION);
            monitor.register("test.ms",FlexMetricType.TIMER);
            var bytes=monitor.bind("test.bytes",FlexMetricTags.of("service","cache"));
            bytes.accept(10);bytes.accept(20);
            assertEquals(2,registry.find("flexlb.test.bytes").summary().count());
            assertEquals(30,registry.find("flexlb.test.bytes").summary().totalAmount());
            var time=monitor.bind("test.ms",FlexMetricTags.of());time.accept(0.25);
            assertEquals(0.25,registry.find("flexlb.test.ms").timer().totalTime(TimeUnit.MILLISECONDS),0.0001);
        } finally { registry.close(); }
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
