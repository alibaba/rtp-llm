package org.flexlb.metric;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class SafeFlexMonitorTest {

    static Stream<Object[]> operations() {
        var tags = FlexMetricTags.of("role", "PREFILL");
        var operations = java.util.List.<Consumer<FlexMonitor>>of(
                monitor -> monitor.register("metric", FlexMetricType.GAUGE),
                monitor -> monitor.register("metric", FlexMetricType.TIMER, FlexPriorityType.PRECISE),
                monitor -> monitor.register("metric", FlexMetricType.QPS, 17),
                monitor -> monitor.report("metric", 2),
                monitor -> monitor.report("metric", tags, 2),
                monitor -> monitor.prepare("metric", tags),
                FlexMonitor::close);
        return operations.stream().flatMap(operation -> Stream.of(
                new Object[]{operation, false}, new Object[]{operation, true}));
    }

    @ParameterizedTest
    @MethodSource("operations")
    void failedOperationDoesNotEscapeOrDisableLaterOperations(Consumer<FlexMonitor> operation, boolean error) {
        var failNext = new AtomicBoolean(true);
        var calls = new AtomicInteger();
        var delegate = mock(FlexMonitor.class, invocation -> {
            calls.incrementAndGet();
            if (failNext.getAndSet(false)) {
                if (error) {
                    throw new AssertionError("monitor failed");
                }
                throw new IllegalStateException("monitor failed");
            }
            return null;
        });
        var monitor = SafeFlexMonitor.wrap(delegate);
        assertDoesNotThrow(() -> operation.accept(monitor));
        assertDoesNotThrow(() -> operation.accept(monitor));
        assertEquals(2, calls.get());
    }

    @Test
    void successfulOperationsPreserveBackendArguments() {
        var delegate = mock(FlexMonitor.class);
        var monitor = SafeFlexMonitor.wrap(delegate);
        var tags = FlexMetricTags.of("role", "PREFILL");
        monitor.register("gauge", FlexMetricType.GAUGE);
        monitor.register("timer", FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register("qps", FlexMetricType.QPS, 17);
        monitor.report("gauge", 2);
        monitor.report("timer", tags, 3);
        monitor.prepare("timer", tags);
        monitor.close();
        verify(delegate).register("gauge", FlexMetricType.GAUGE);
        verify(delegate).register("timer", FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        verify(delegate).register("qps", FlexMetricType.QPS, 17);
        verify(delegate).report("gauge", 2);
        verify(delegate).report("timer", tags, 3);
        verify(delegate).prepare("timer", tags);
        verify(delegate).close();
        verifyNoMoreInteractions(delegate);
    }

    @Test
    void wrappingIsIdempotentAndMissingMonitorIsNoOp() {
        var monitor = SafeFlexMonitor.wrap(mock(FlexMonitor.class));
        assertSame(monitor, SafeFlexMonitor.wrap(monitor));
        assertSame(NoOpFlexMonitor.getInstance(), SafeFlexMonitor.wrap(null));
        assertSame(NoOpFlexMonitor.getInstance(), SafeFlexMonitor.wrap(NoOpFlexMonitor.getInstance()));
    }

    @Test
    void diagnosticLoggingFailureCannotEscapeMonitoringBoundary() {
        var delegate = mock(FlexMonitor.class);
        doThrow(new IllegalStateException("monitor failed")).when(delegate).report("broken", 1);
        var monitor = SafeFlexMonitor.wrap(delegate);
        var logger = (Logger) LoggerFactory.getLogger(SafeFlexMonitor.class);
        Level previousLevel = logger.getLevel();
        var loggingCalls = new AtomicInteger();
        var appender = new AppenderBase<ILoggingEvent>() {
            @Override
            protected void append(ILoggingEvent event) {
                loggingCalls.incrementAndGet();
                throw new AssertionError("logger failed");
            }
        };
        appender.start();
        logger.addAppender(appender);
        logger.setLevel(Level.DEBUG);
        try {
            assertDoesNotThrow(() -> monitor.report("broken", 1));
            assertEquals(1, loggingCalls.get());
            monitor.report("healthy", 2);
            verify(delegate).report("healthy", 2);
        } finally {
            logger.detachAppender(appender);
            appender.stop();
            logger.setLevel(previousLevel);
        }
    }
}
