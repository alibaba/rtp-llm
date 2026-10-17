package org.flexlb.metric;

import io.micrometer.core.instrument.Timer;
import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import org.flexlb.enums.FlexMetricType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class MicrometerFlexMonitorTest {

    @AfterEach
    void resetAllowedMetrics() {
        MicrometerFlexMonitor.setAllowedMetrics(null);
    }

    @ParameterizedTest
    @CsvSource({"GAUGE,false", "GAUGE,true", "COUNTER,false", "COUNTER,true", "TIMER,false", "TIMER,true"})
    void backendFailureDropsOnlyBrokenMetric(FlexMetricType type, boolean error) {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        try {
            registry.config().meterFilter(new MeterFilter() {
                @Override
                public Meter.Id map(Meter.Id id) {
                    if (id.getName().equals("flexlb.broken")) {
                        if (error) {
                            throw new AssertionError("monitor failed");
                        }
                        throw new IllegalStateException("monitor failed");
                    }
                    return id;
                }
            });
            var monitor = new MicrometerFlexMonitor(registry);
            monitor.register("broken", type);
            monitor.register("healthy", type);
            assertDoesNotThrow(() -> monitor.prepare("broken", null));
            assertDoesNotThrow(() -> monitor.report("broken", 3));
            monitor.report("healthy", 7);
            switch (type) {
                case GAUGE -> assertEquals(7, registry.get("flexlb.healthy").gauge().value());
                case COUNTER -> assertEquals(7, registry.get("flexlb.healthy").counter().count());
                case TIMER -> assertEquals(7, registry.get("flexlb.healthy").timer().totalTime(TimeUnit.MILLISECONDS));
                default -> throw new AssertionError(type);
            }
        } finally {
            registry.close();
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void brokenTagsDoNotEscapeReporting(boolean error) {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            var tags = mock(FlexMetricTags.class);
            if (error) {
                when(tags.isEmpty()).thenThrow(new AssertionError("bad tags"));
            } else {
                when(tags.isEmpty()).thenThrow(new IllegalStateException("bad tags"));
            }
            assertDoesNotThrow(() -> monitor.prepare("broken.tags", tags));
            assertDoesNotThrow(() -> monitor.report("broken.tags", tags, 1));
            monitor.report("healthy", 2);
            assertEquals(2, registry.get("flexlb.healthy").gauge().value());
        } finally {
            registry.close();
        }
    }

    @Test
    void invalidRegistrationAndLookupDoNotEscape() {
        SimpleMeterRegistry registry = new SimpleMeterRegistry();
        try {
            var monitor = new MicrometerFlexMonitor(registry);
            assertDoesNotThrow(() -> monitor.register(null, FlexMetricType.GAUGE));
            assertDoesNotThrow(() -> monitor.register("invalid", null));
            assertDoesNotThrow(() -> monitor.prepare(null, null));
            assertDoesNotThrow(() -> monitor.report(null, 1));
            monitor.report("healthy", 2);
            assertEquals(2, registry.get("flexlb.healthy").gauge().value());
        } finally {
            registry.close();
        }
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
