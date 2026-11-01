package org.flexlb.config;

import io.micrometer.core.instrument.simple.SimpleMeterRegistry;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.SafeFlexMonitor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import javax.annotation.PostConstruct;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.fail;

class MonitorClientConfigTest {

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void customSpringMonitorIsProtectedWithoutLosingBackendInitialization(boolean error) {
        var backend = new BrokenMonitor(error);
        try (var context = new AnnotationConfigApplicationContext()) {
            context.register(MonitorClientConfig.class);
            context.registerBean("customMonitor", FlexMonitor.class, () -> backend);
            context.registerBean(Consumer.class);
            assertDoesNotThrow(context::refresh);
            assertTrue(backend.initialized);
            var monitor = context.getBean(FlexMonitor.class);
            assertInstanceOf(SafeFlexMonitor.class, monitor);
            assertDoesNotThrow(() -> monitor.prepare("metric", null));
            assertDoesNotThrow(monitor::close);
            assertTrue(backend.closed);
        }
    }

    @Test
    void pullMetricsRemainLiveAlongsideAnIndependentBusinessBackend() {
        var backend = org.mockito.Mockito.mock(FlexMonitor.class);
        var source = new AtomicLong(7);
        var registry = new SimpleMeterRegistry();
        try (var context = new AnnotationConfigApplicationContext()) {
            context.register(MonitorClientConfig.class);
            context.registerBean("businessMonitor", FlexMonitor.class, () -> backend);
            context.registerBean(
                    "meterRegistry",
                    io.micrometer.core.instrument.MeterRegistry.class,
                    () -> registry);
            context.refresh();
            var monitor = context.getBean(FlexMonitor.class);
            monitor.observe(
                    "delivery_executor_active_threads",
                    FlexMetricType.GAUGE,
                    source,
                    AtomicLong::doubleValue);
            monitor.observe(
                    "delivery_executor_completed_tasks",
                    FlexMetricType.COUNTER,
                    source,
                    AtomicLong::doubleValue);
            monitor.report("business", 3);
            org.mockito.Mockito.verify(backend).report("business", 3);
            assertEquals(
                    7, registry.get("flexlb.delivery_executor_active_threads").gauge().value());
            assertEquals(
                    7,
                    registry.get("flexlb.delivery_executor_completed_tasks")
                            .functionCounter()
                            .count());
            source.set(19);
            assertEquals(
                    19, registry.get("flexlb.delivery_executor_active_threads").gauge().value());
            assertEquals(
                    19,
                    registry.get("flexlb.delivery_executor_completed_tasks")
                            .functionCounter()
                            .count());
            org.mockito.Mockito.verify(backend, org.mockito.Mockito.never())
                    .observe(
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any(),
                            org.mockito.ArgumentMatchers.any());
        }
    }

    static class Consumer {
        Consumer(FlexMonitor monitor) {
            monitor.register("metric", FlexMetricType.GAUGE);
            monitor.report("metric", 1);
        }
    }

    static class BrokenMonitor implements FlexMonitor {
        private final boolean error;
        boolean initialized;
        boolean closed;

        BrokenMonitor(boolean error) {
            this.error = error;
        }

        @PostConstruct
        void init() {
            initialized = true;
        }

        private void fail() {
            if (error) {
                throw new AssertionError("monitor failed");
            }
            throw new IllegalStateException("monitor failed");
        }

        public void register(String name, FlexMetricType type) {
            fail();
        }

        public void register(String name, FlexMetricType type, FlexPriorityType priority) {
            fail();
        }

        public void register(String name, FlexMetricType type, int statisticsType) {
            fail();
        }

        public void report(String name, double value) {
            fail();
        }

        public void report(String name, FlexMetricTags tags, double value) {
            fail();
        }

        public void prepare(String name, FlexMetricTags tags) {
            fail();
        }

        public void close() {
            closed = true;
            fail();
        }
    }
}
