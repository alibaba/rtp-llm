package org.flexlb.engine.grpc.monitor;

import io.netty.buffer.PooledByteBufAllocator;
import org.flexlb.config.ConfigService;
import org.flexlb.config.NettyMemoryReporterConfig;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.lang.management.BufferPoolMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.LongSupplier;

import static org.flexlb.constant.MetricConstant.NETTY_DIRECT_PINNED_BYTES;
import static org.flexlb.constant.MetricConstant.NETTY_DIRECT_RESERVED_BYTES;
import static org.flexlb.constant.MetricConstant.NETTY_MEMORY_SAMPLE_DURATION_MS;
import static org.flexlb.constant.MetricConstant.NETTY_MEMORY_SAMPLE_TIMESTAMP_MS;

/**
 * Opt-in, fixed-delay allocator scans. Netty 4.1.101's getters take arena locks
 * and traverse chunks, so neither RPC threads nor metric scrapes call them.
 */
@Component
public class NettyMemoryReporter {
    private static final Logger logger = LoggerFactory.getLogger(NettyMemoryReporter.class);
    private static final FlexMetricTags NO_TAGS = FlexMetricTags.of();
    private final NettyMemoryReporterConfig settings;
    private final FlexMonitor monitor;
    private final LongSupplier nanoTime;
    private final List<AllocatorSample> allocators = new ArrayList<>(2);
    private ScheduledExecutorService sampler;
    private BufferPoolMXBean directPool;
    private Long lastCaptureNanos;

    @Autowired
    public NettyMemoryReporter(ConfigService configService, FlexMonitor monitor) {
        this(configService.loadBalanceConfig().getScheduler().getNettyMemoryReporter(), monitor);
    }

    NettyMemoryReporter(NettyMemoryReporterConfig settings, FlexMonitor monitor) {
        this(settings, monitor, System::nanoTime);
    }

    NettyMemoryReporter(
            NettyMemoryReporterConfig settings, FlexMonitor monitor, LongSupplier nanoTime) {
        this.settings = settings;
        this.monitor = monitor;
        this.nanoTime = nanoTime;
    }

    @PostConstruct
    public void start() {
        // Do not initialize allocators, register meters, or create a task while disabled.
        if (!settings.isEnabled()) {
            return;
        }
        try {
            discoverAllocators();
            if (settings.getAutoCapture().isEnabled()) {
                directPool =
                        ManagementFactory.getPlatformMXBeans(BufferPoolMXBean.class).stream()
                                .filter(pool -> "direct".equals(pool.getName()))
                                .findFirst()
                                .orElse(null);
            }
            monitor.register(
                    NETTY_DIRECT_RESERVED_BYTES, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
            monitor.register(
                    NETTY_DIRECT_PINNED_BYTES, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
            monitor.register(
                    NETTY_MEMORY_SAMPLE_DURATION_MS,
                    FlexMetricType.GAUGE,
                    FlexPriorityType.PRECISE);
            monitor.register(
                    NETTY_MEMORY_SAMPLE_TIMESTAMP_MS,
                    FlexMetricType.GAUGE,
                    FlexPriorityType.PRECISE);
            for (AllocatorSample sample : allocators) {
                monitor.prepare(NETTY_DIRECT_RESERVED_BYTES, sample.tags);
                monitor.prepare(NETTY_DIRECT_PINNED_BYTES, sample.tags);
            }
            monitor.prepare(NETTY_MEMORY_SAMPLE_DURATION_MS, NO_TAGS);
            monitor.prepare(NETTY_MEMORY_SAMPLE_TIMESTAMP_MS, NO_TAGS);
            sampler =
                    Executors.newSingleThreadScheduledExecutor(
                            task -> {
                                Thread thread = new Thread(task, "flexlb-netty-memory-reporter");
                                thread.setDaemon(true);
                                return thread;
                            });
            sampler.scheduleWithFixedDelay(
                    this::sampleSafely, 0L, settings.getIntervalMs(), TimeUnit.MILLISECONDS);
        } catch (RuntimeException failure) {
            close();
            logger.warn(
                    "event=netty_memory_reporter_start_failed type={}",
                    failure.getClass().getName());
        }
    }

    void discoverAllocators() {
        // EngineGrpcClient explicitly uses this pool. Default HTTP channels may share it.
        registerAllocator("netty_default", PooledByteBufAllocator.DEFAULT);
        try {
            // gRPC 1.65.1 uses a separate lazy singleton (normally 2 MiB chunks).
            // Its public getter lives on a package-private class. Read that getter
            // once without replacing the transport allocator or changing its settings.
            var getter =
                    Class.forName("io.grpc.netty.Utils")
                            .getDeclaredMethod("getByteBufAllocator", boolean.class);
            if (!getter.trySetAccessible()) {
                throw new IllegalAccessException("gRPC allocator getter is inaccessible");
            }
            Object allocator = getter.invoke(null, false);
            if (allocator instanceof PooledByteBufAllocator pooled) {
                registerAllocator("grpc_default", pooled);
            } else {
                logger.warn(
                        "event=netty_memory_allocator_unavailable source=grpc_default reason=not_pooled");
            }
        } catch (ReflectiveOperationException | RuntimeException failure) {
            logger.warn(
                    "event=netty_memory_allocator_unavailable source=grpc_default type={}",
                    failure.getClass().getName());
        }
    }

    void registerAllocator(String name, PooledByteBufAllocator allocator) {
        for (AllocatorSample existing : allocators) {
            if (existing.allocator == allocator) {
                logger.info(
                        "event=netty_memory_allocator_shared source={} allocator={}",
                        name,
                        existing.name);
                return;
            }
        }
        allocators.add(new AllocatorSample(name, allocator));
    }

    private void sampleSafely() {
        try {
            sample();
        } catch (OutOfMemoryError ignored) {
            // Do not allocate diagnostics while the process cannot allocate.
        } catch (RuntimeException failure) {
            logger.warn("event=netty_memory_sample_failed type={}", failure.getClass().getName());
        }
    }

    void sample() {
        if (!settings.isEnabled()) {
            return;
        }
        long started = nanoTime.getAsLong();
        long totalReserved = 0;
        for (AllocatorSample sample : allocators) {
            sample.reserved = sample.allocator.metric().usedDirectMemory();
            sample.pinned = sample.allocator.pinnedDirectMemory();
            totalReserved += Math.max(0L, sample.reserved);
            monitor.report(NETTY_DIRECT_RESERVED_BYTES, sample.tags, sample.reserved);
            monitor.report(NETTY_DIRECT_PINNED_BYTES, sample.tags, sample.pinned);
        }
        double durationMs = (nanoTime.getAsLong() - started) / 1_000_000.0;
        monitor.report(NETTY_MEMORY_SAMPLE_DURATION_MS, NO_TAGS, durationMs);
        // A stale timestamp distinguishes old cached values from a fresh allocator scan.
        monitor.report(NETTY_MEMORY_SAMPLE_TIMESTAMP_MS, NO_TAGS, System.currentTimeMillis());
        captureIfNeeded(totalReserved, durationMs);
    }

    private void captureIfNeeded(long totalReserved, double durationMs) {
        var capture = settings.getAutoCapture();
        if (!capture.isEnabled()
                || (lastCaptureNanos != null
                        && nanoTime.getAsLong() - lastCaptureNanos
                                < TimeUnit.MILLISECONDS.toNanos(capture.getCooldownMs()))) {
            return;
        }
        long jvmDirectBytes = directPool == null ? -1L : directPool.getMemoryUsed();
        // JVM BufferPool counters exclude some no-cleaner allocations; keep both views.
        if (Math.max(jvmDirectBytes, totalReserved) < capture.getThresholdBytes()) {
            return;
        }
        // Reuse only this scan's values. No extra scan, stack dump, JFR, or heap dump.
        for (AllocatorSample sample : allocators) {
            logger.warn(
                    "event=netty_memory_capture allocator={} reservedBytes={} pinnedBytes={} "
                            + "jvmDirectBytes={} sampleDurationMs={}",
                    sample.name,
                    sample.reserved,
                    sample.pinned,
                    jvmDirectBytes,
                    durationMs);
        }
        lastCaptureNanos = nanoTime.getAsLong();
    }

    @PreDestroy
    public void close() {
        if (sampler != null) {
            sampler.shutdownNow();
        }
    }

    private static final class AllocatorSample {
        private final String name;
        private final PooledByteBufAllocator allocator;
        private final FlexMetricTags tags;
        private long reserved;
        private long pinned;

        private AllocatorSample(String name, PooledByteBufAllocator allocator) {
            this.name = name;
            this.allocator = allocator;
            this.tags = FlexMetricTags.of("allocator", name);
        }
    }
}
