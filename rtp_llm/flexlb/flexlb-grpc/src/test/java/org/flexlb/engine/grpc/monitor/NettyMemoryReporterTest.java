package org.flexlb.engine.grpc.monitor;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.PooledByteBufAllocator;
import org.flexlb.config.NettyMemoryReporterConfig;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.constant.MetricConstant.NETTY_DIRECT_PINNED_BYTES;
import static org.flexlb.constant.MetricConstant.NETTY_DIRECT_RESERVED_BYTES;
import static org.flexlb.constant.MetricConstant.NETTY_MEMORY_SAMPLE_DURATION_MS;
import static org.flexlb.constant.MetricConstant.NETTY_MEMORY_SAMPLE_TIMESTAMP_MS;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.clearInvocations;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;

class NettyMemoryReporterTest {
    @Test
    void disabledStartupAndSamplingNeverDiscoverScanOrReport() {
        var settings = new NettyMemoryReporterConfig();
        var monitor = mock(FlexMonitor.class);
        var reporter = spy(new NettyMemoryReporter(settings, monitor));
        var allocator = mock(PooledByteBufAllocator.class);
        reporter.registerAllocator("test", allocator);
        reporter.start();
        reporter.sample();
        reporter.close();
        verify(reporter, never()).discoverAllocators();
        verifyNoInteractions(allocator, monitor);
    }

    @Test
    void realBufferAllocationAndReleaseChangePinnedMetricWithoutDoubleCountingSharedPool() {
        var settings = enabled();
        var monitor = mock(FlexMonitor.class);
        var reporter = new NettyMemoryReporter(settings, monitor);
        // Small private pool, no thread cache: the test observes actual Netty accounting.
        var allocator = new PooledByteBufAllocator(true, 0, 1, 8192, 4, 0, 0, false);
        reporter.registerAllocator("test", allocator);
        reporter.registerAllocator("shared_alias", allocator);
        var tags = FlexMetricTags.of("allocator", "test");
        ByteBuf buffer = allocator.directBuffer(16384);
        try {
            reporter.sample();
            verify(monitor)
                    .report(
                            NETTY_DIRECT_PINNED_BYTES,
                            tags,
                            (double) allocator.pinnedDirectMemory());
            verify(monitor)
                    .report(
                            NETTY_DIRECT_RESERVED_BYTES,
                            tags,
                            (double) allocator.metric().usedDirectMemory());
            assertTrue(allocator.pinnedDirectMemory() >= 16384L);
            verify(monitor, times(1)).report(eq(NETTY_DIRECT_PINNED_BYTES), any(), anyDouble());
            clearInvocations(monitor);
        } finally {
            buffer.release();
        }
        reporter.sample();
        verify(monitor).report(NETTY_DIRECT_PINNED_BYTES, tags, 0.0);
    }

    @Test
    void discoveryIncludesTheActualGrpcSingletonAndNettyDefault() throws Exception {
        var getter =
                Class.forName("io.grpc.netty.Utils")
                        .getDeclaredMethod("getByteBufAllocator", boolean.class);
        assertTrue(getter.trySetAccessible());
        var grpc = (PooledByteBufAllocator) getter.invoke(null, false);
        var monitor = mock(FlexMonitor.class);
        var reporter = new NettyMemoryReporter(enabled(), monitor);
        reporter.discoverAllocators();
        ByteBuf buffer = grpc.directBuffer(32768);
        try {
            reporter.sample();
            var name = grpc == PooledByteBufAllocator.DEFAULT ? "netty_default" : "grpc_default";
            verify(monitor)
                    .report(
                            NETTY_DIRECT_PINNED_BYTES,
                            FlexMetricTags.of("allocator", name),
                            (double) grpc.pinnedDirectMemory());
        } finally {
            buffer.release();
        }
    }

    @Test
    void enabledLifecycleSamplesOnDedicatedDaemonAndShutsDown() throws Exception {
        var settings = enabled();
        var monitor = mock(FlexMonitor.class);
        var sampled = new CountDownLatch(1);
        var samplerThread = new java.util.concurrent.atomic.AtomicReference<Thread>();
        doAnswer(
                        invocation -> {
                            samplerThread.set(Thread.currentThread());
                            sampled.countDown();
                            return null;
                        })
                .when(monitor)
                .report(eq(NETTY_MEMORY_SAMPLE_TIMESTAMP_MS), any(), anyDouble());
        var reporter = new NettyMemoryReporter(settings, monitor);
        try {
            reporter.start();
            verify(monitor)
                    .register(
                            NETTY_DIRECT_RESERVED_BYTES,
                            FlexMetricType.GAUGE,
                            FlexPriorityType.PRECISE);
            verify(monitor)
                    .register(
                            NETTY_DIRECT_PINNED_BYTES,
                            FlexMetricType.GAUGE,
                            FlexPriorityType.PRECISE);
            verify(monitor)
                    .register(
                            NETTY_MEMORY_SAMPLE_DURATION_MS,
                            FlexMetricType.GAUGE,
                            FlexPriorityType.PRECISE);
            verify(monitor)
                    .register(
                            NETTY_MEMORY_SAMPLE_TIMESTAMP_MS,
                            FlexMetricType.GAUGE,
                            FlexPriorityType.PRECISE);
            assertTrue(sampled.await(5, TimeUnit.SECONDS));
            assertTrue(samplerThread.get().isDaemon());
            assertEquals("flexlb-netty-memory-reporter", samplerThread.get().getName());
            assertNotSame(Thread.currentThread(), samplerThread.get());
        } finally {
            reporter.close();
        }
        samplerThread.get().join(5000L);
        assertFalse(samplerThread.get().isAlive(), "shutdown must release the diagnostic thread");
    }

    @Test
    void samplingContinuesAfterAnOutOfMemoryErrorWithoutLoggingAnotherFailure() throws Exception {
        var settings = enabled();
        settings.setIntervalMs(1000L);
        var monitor = mock(FlexMonitor.class);
        var attempts = new AtomicInteger();
        var recovered = new CountDownLatch(1);
        doAnswer(
                        invocation -> {
                            if (attempts.incrementAndGet() == 1) {
                                throw new OutOfMemoryError("simulated reporting failure");
                            }
                            recovered.countDown();
                            return null;
                        })
                .when(monitor)
                .report(eq(NETTY_DIRECT_RESERVED_BYTES), any(), anyDouble());
        Logger logger = (Logger) LoggerFactory.getLogger(NettyMemoryReporter.class);
        var logs = new ListAppender<ILoggingEvent>();
        logs.start();
        logger.addAppender(logs);
        var reporter = new NettyMemoryReporter(settings, monitor);
        try {
            reporter.start();
            assertTrue(
                    recovered.await(5, TimeUnit.SECONDS),
                    "one failed sample must not cancel future sampling");
            assertFalse(
                    logs.list.stream()
                            .anyMatch(
                                    event ->
                                            event.getMessage()
                                                    .startsWith(
                                                            "event=netty_memory_sample_failed")));
        } finally {
            reporter.close();
            logger.detachAppender(logs);
            logs.stop();
        }
    }

    @Test
    void captureIsSeparatelyDisabledAndReusesSamplesWithCooldown() {
        var settings = enabled();
        settings.getAutoCapture().setThresholdBytes(1L);
        settings.getAutoCapture().setCooldownMs(60_000L);
        var nowNanos = new AtomicLong();
        var reporter = new NettyMemoryReporter(settings, mock(FlexMonitor.class), nowNanos::get);
        var allocator = spy(new PooledByteBufAllocator(true, 0, 1, 8192, 4, 0, 0, false));
        reporter.registerAllocator("test", allocator);
        Logger logger = (Logger) LoggerFactory.getLogger(NettyMemoryReporter.class);
        var logs = new ListAppender<ILoggingEvent>();
        logs.start();
        logger.addAppender(logs);
        ByteBuf buffer = allocator.directBuffer(16384);
        clearInvocations(allocator);
        try {
            reporter.sample();
            assertEquals(0L, captures(logs));
            settings.getAutoCapture().setEnabled(true);
            reporter.sample();
            nowNanos.set(TimeUnit.SECONDS.toNanos(10));
            reporter.sample();
            assertEquals(1L, captures(logs));
            nowNanos.set(TimeUnit.SECONDS.toNanos(60));
            reporter.sample();
            assertEquals(2L, captures(logs));
            // Capture does not rescan an allocator in addition to the four periodic scans.
            verify(allocator, times(4)).pinnedDirectMemory();
        } finally {
            buffer.release();
            logger.detachAppender(logs);
            logs.stop();
        }
    }

    @Test
    void cooldownStartsAfterSlowSamplingAndCaptureWhileMetricsContinue() {
        var settings = enabled();
        settings.getAutoCapture().setEnabled(true);
        settings.getAutoCapture().setThresholdBytes(1L);
        settings.getAutoCapture().setCooldownMs(2000L);
        var nowNanos = new AtomicLong();
        var monitor = mock(FlexMonitor.class);
        doAnswer(
                        invocation -> {
                            nowNanos.addAndGet(TimeUnit.MILLISECONDS.toNanos(1500));
                            return null;
                        })
                .doNothing()
                .when(monitor)
                .report(eq(NETTY_DIRECT_RESERVED_BYTES), any(), anyDouble());
        var reporter = new NettyMemoryReporter(settings, monitor, nowNanos::get);
        var allocator = new PooledByteBufAllocator(true, 0, 1, 8192, 4, 0, 0, false);
        reporter.registerAllocator("test", allocator);
        Logger logger = (Logger) LoggerFactory.getLogger(NettyMemoryReporter.class);
        var logs =
                new ListAppender<ILoggingEvent>() {
                    @Override
                    protected void append(ILoggingEvent event) {
                        super.append(event);
                        if (event.getMessage().startsWith("event=netty_memory_capture ")
                                && captures(this) == 1L) {
                            nowNanos.addAndGet(TimeUnit.MILLISECONDS.toNanos(500));
                        }
                    }
                };
        logs.start();
        logger.addAppender(logs);
        ByteBuf buffer = allocator.directBuffer(16384);
        try {
            reporter.sample();
            assertEquals(TimeUnit.SECONDS.toNanos(2), nowNanos.get());
            assertEquals(1L, captures(logs));

            nowNanos.set(TimeUnit.SECONDS.toNanos(3));
            reporter.sample();
            assertEquals(1L, captures(logs), "slow sampling must not consume the capture cooldown");

            nowNanos.set(TimeUnit.SECONDS.toNanos(4) - 1L);
            reporter.sample();
            assertEquals(1L, captures(logs), "cooldown starts after logging finishes");

            nowNanos.incrementAndGet();
            reporter.sample();
            assertEquals(2L, captures(logs), "capture resumes exactly at the cooldown boundary");
            verify(monitor, times(4)).report(eq(NETTY_DIRECT_RESERVED_BYTES), any(), anyDouble());
            verify(monitor, times(4)).report(eq(NETTY_DIRECT_PINNED_BYTES), any(), anyDouble());
            verify(monitor, times(4))
                    .report(eq(NETTY_MEMORY_SAMPLE_DURATION_MS), any(), anyDouble());
            verify(monitor, times(4))
                    .report(eq(NETTY_MEMORY_SAMPLE_TIMESTAMP_MS), any(), anyDouble());
        } finally {
            buffer.release();
            logger.detachAppender(logs);
            logs.stop();
        }
    }

    private static long captures(ListAppender<ILoggingEvent> logs) {
        return logs.list.stream()
                .filter(event -> event.getMessage().startsWith("event=netty_memory_capture "))
                .count();
    }

    private static NettyMemoryReporterConfig enabled() {
        var config = new NettyMemoryReporterConfig();
        config.setEnabled(true);
        return config;
    }
}
