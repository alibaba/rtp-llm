package org.flexlb.metric;

import com.google.common.util.concurrent.AtomicDouble;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

import javax.annotation.PreDestroy;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * MicrometerFlexMonitor - bridges FlexMonitor interface to micrometer MeterRegistry.
 *
 * <p>This implementation allows FlexLB business metrics to be exposed via the
 * Spring Boot Actuator {@code /prometheus} endpoint in environments where
 * kmonitor is not available (e.g. 110 test environment built with {@code -P '!internal'}).
 *
 * <p>Metric type mapping:
 * <ul>
 *   <li>{@link FlexMetricType#QPS} / {@link FlexMetricType#COUNTER} → micrometer Counter (increment)</li>
 *   <li>{@link FlexMetricType#GAUGE} → micrometer Gauge (absolute value via AtomicDouble)</li>
 *   <li>{@link FlexMetricType#TIMER} → micrometer Timer (duration distribution with p50/p90/p95/p99)</li>
 *   <li>Unknown types default to Gauge</li>
 * </ul>
 *
 * <p>All metric names are prefixed with {@code flexlb.} to avoid conflicts with
 * Spring Boot's built-in metrics.
 */
@Slf4j
public class MicrometerFlexMonitor implements FlexMonitor {

    private static final String METRIC_PREFIX = "flexlb.";
    private static final long DEFAULT_RETIREMENT_SWEEP_INTERVAL_MS = 1_000L;
    private static final String ROLE_TAG = "role";
    private static final String ENGINE_IP_TAG = "engineIp";
    private static final String ENGINE_IP_PORT_TAG = "engineIpPort";
    private static final int REGISTRATION_LOCK_COUNT = 64;

    /**
     * When non-null, only metrics whose names are in this set will be registered/reported.
     * Set by {@link org.flexlb.config.CriticalMetricsFilterConfig} when
     * {@code flexlb.monitor.mode=critical-only} is active.
     */
    private static volatile Set<String> ALLOWED_METRICS = null;

    /**
     * Sets the allowlist of metric names (without the {@code flexlb.} prefix).
     * Pass {@code null} to disable filtering (allow all metrics).
     */
    public static void setAllowedMetrics(Set<String> metrics) {
        ALLOWED_METRICS = metrics;
        log.info("MicrometerFlexMonitor allowlist set: {} metrics", metrics == null ? "unlimited" : metrics.size());
    }

    private final MeterRegistry meterRegistry;
    private final ConcurrentHashMap<String, FlexMetricType> metricTypes = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<MetricKey, GaugeEntry> gaugeCache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<MetricKey, Counter> counterCache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<MetricKey, Timer> timerCache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<FlexMetricTags, Tags> tagsCache = new ConcurrentHashMap<>();
    private final Map<EndpointScopeKey, EndpointScopeState> endpointScopes = new HashMap<>();
    private final ReentrantReadWriteLock registrationLifecycleLock = new ReentrantReadWriteLock();
    private final Object endpointScopeLock = new Object();
    private final Object[] registrationLocks = new Object[REGISTRATION_LOCK_COUNT];
    private final ScheduledExecutorService retirementExecutor;
    private final AtomicBoolean closed = new AtomicBoolean();

    /**
     * Constructor.
     *
     * @param meterRegistry micrometer MeterRegistry (typically PrometheusMeterRegistry)
     */
    public MicrometerFlexMonitor(MeterRegistry meterRegistry) {
        this(meterRegistry, DEFAULT_RETIREMENT_SWEEP_INTERVAL_MS);
    }

    MicrometerFlexMonitor(MeterRegistry meterRegistry, long retirementSweepIntervalMs) {
        this.meterRegistry = meterRegistry;
        for (int i = 0; i < registrationLocks.length; i++) {
            registrationLocks[i] = new Object();
        }
        AtomicInteger threadNumber = new AtomicInteger();
        ThreadFactory threadFactory = runnable -> {
            Thread thread = new Thread(runnable,
                    "flexlb-metric-retirement-" + threadNumber.incrementAndGet());
            thread.setDaemon(true);
            return thread;
        };
        retirementExecutor = Executors.newSingleThreadScheduledExecutor(threadFactory);
        long sweepIntervalMs = Math.max(1L, retirementSweepIntervalMs);
        retirementExecutor.scheduleWithFixedDelay(
                this::runRetirementSweepSafely,
                sweepIntervalMs,
                sweepIntervalMs,
                TimeUnit.MILLISECONDS);
        log.info("MicrometerFlexMonitor initialized with MeterRegistry: {}",
                meterRegistry.getClass().getSimpleName());
    }

    @Override
    public void register(String metricName, FlexMetricType metricType) {
        if (ALLOWED_METRICS != null && !ALLOWED_METRICS.contains(metricName)) {
            return;
        }
        metricTypes.put(metricName, metricType);
        log.debug("Registered metric: {} as type {}", metricName, metricType);
    }

    @Override
    public void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType) {
        register(metricName, metricType);
    }

    @Override
    public void register(String metricName, FlexMetricType metricType, int statisticsType) {
        register(metricName, metricType);
    }

    @Override
    public void report(String metricName, double value) {
        report(metricName, null, value);
    }

    @Override
    public void report(String metricName, FlexMetricTags metricsTags, double value) {
        if (ALLOWED_METRICS != null && !ALLOWED_METRICS.contains(metricName)) {
            return;
        }
        String prefixedName = METRIC_PREFIX + metricName;
        FlexMetricType metricType = metricTypes.getOrDefault(metricName, FlexMetricType.GAUGE);
        MetricKey key = new MetricKey(prefixedName, metricsTags);

        try {
            switch (metricType) {
                case QPS:
                case COUNTER:
                    Counter counter = counter(key);
                    if (counter != null) {
                        counter.increment(value);
                    }
                    break;
                case TIMER:
                    Timer timer = timer(key);
                    if (timer != null) {
                        timer.record((long) value, TimeUnit.MILLISECONDS);
                    }
                    break;
                case GAUGE:
                default:
                    reportGauge(key, value);
                    break;
            }
        } catch (Exception e) {
            log.warn("Failed to report metric {}: {}", metricName, e.getMessage());
        }
    }

    @Override
    public void prepare(String metricName, FlexMetricTags metricsTags) {
        if (ALLOWED_METRICS != null && !ALLOWED_METRICS.contains(metricName)) {
            return;
        }
        String prefixedName = METRIC_PREFIX + metricName;
        FlexMetricType metricType = metricTypes.getOrDefault(metricName, FlexMetricType.GAUGE);
        MetricKey key = new MetricKey(prefixedName, metricsTags);
        try {
            switch (metricType) {
                case QPS:
                case COUNTER:
                    counter(key);
                    break;
                case TIMER:
                    timer(key);
                    break;
                case GAUGE:
                default:
                    gaugeEntry(key);
                    break;
            }
        } catch (Exception e) {
            log.warn("Failed to prepare metric {}: {}", metricName, e.getMessage());
        }
    }

    private Tags resolveTags(FlexMetricTags metricsTags) {
        return (metricsTags == null || metricsTags.isEmpty())
                ? Tags.empty()
                : tagsCache.computeIfAbsent(metricsTags, this::toMicrometerTags);
    }

    @Override
    public MetricLease acquireEndpointScope(FlexMetricTags endpointTags, long retirementGraceMs) {
        EndpointScopeKey scopeKey = EndpointScopeKey.from(endpointTags);
        long graceMs = Math.max(0L, retirementGraceMs);

        registrationLifecycleLock.writeLock().lock();
        try {
            synchronized (endpointScopeLock) {
                EndpointScopeState state = endpointScopes.computeIfAbsent(
                        scopeKey, ignored -> new EndpointScopeState());
                state.references++;
                state.retirementGraceMs = Math.max(state.retirementGraceMs, graceMs);
                state.retireAtMs = Long.MAX_VALUE;
                state.purgeAtMs = Long.MAX_VALUE;
                state.cleaned = false;
            }
        } finally {
            registrationLifecycleLock.writeLock().unlock();
        }

        AtomicBoolean released = new AtomicBoolean();
        return () -> {
            if (released.compareAndSet(false, true)) {
                releaseEndpointScope(scopeKey);
            }
        };
    }

    private Counter counter(MetricKey key) {
        Counter counter = counterCache.get(key);
        if (counter == null) {
            registrationLifecycleLock.readLock().lock();
            try {
                if (!registrationAllowed(key.metricsTags())) {
                    return null;
                }
                synchronized (registrationLock(key)) {
                    counter = counterCache.get(key);
                    if (counter == null) {
                        counter = Counter.builder(key.name())
                                .tags(resolveTags(key.metricsTags()))
                                .register(meterRegistry);
                        counterCache.put(key, counter);
                    }
                }
            } finally {
                registrationLifecycleLock.readLock().unlock();
            }
        }
        return counter;
    }

    private Timer timer(MetricKey key) {
        Timer timer = timerCache.get(key);
        if (timer == null) {
            registrationLifecycleLock.readLock().lock();
            try {
                if (!registrationAllowed(key.metricsTags())) {
                    return null;
                }
                synchronized (registrationLock(key)) {
                    timer = timerCache.get(key);
                    if (timer == null) {
                        timer = Timer.builder(key.name())
                                .tags(resolveTags(key.metricsTags()))
                                .publishPercentileHistogram(true)
                                .publishPercentiles(0.5, 0.9, 0.95, 0.99)
                                .register(meterRegistry);
                        timerCache.put(key, timer);
                    }
                }
            } finally {
                registrationLifecycleLock.readLock().unlock();
            }
        }
        return timer;
    }

    /**
     * Report a gauge value. Uses an AtomicDouble per metric-name+tags combination
     * as the gauge reference object, so each unique tag set gets its own gauge.
     *
     * @param value absolute gauge value
     */
    private void reportGauge(MetricKey key, double value) {
        GaugeEntry entry = gaugeEntry(key);
        if (entry != null) {
            entry.value().set(value);
        }
    }

    private GaugeEntry gaugeEntry(MetricKey key) {
        GaugeEntry entry = gaugeCache.get(key);
        if (entry == null) {
            registrationLifecycleLock.readLock().lock();
            try {
                if (!registrationAllowed(key.metricsTags())) {
                    return null;
                }
                synchronized (registrationLock(key)) {
                    entry = gaugeCache.get(key);
                    if (entry == null) {
                        AtomicDouble value = new AtomicDouble(0.0);
                        Gauge gauge = Gauge.builder(key.name(), value, AtomicDouble::get)
                                .tags(resolveTags(key.metricsTags()))
                                .register(meterRegistry);
                        entry = new GaugeEntry(value, gauge);
                        gaugeCache.put(key, entry);
                    }
                }
            } finally {
                registrationLifecycleLock.readLock().unlock();
            }
        }
        return entry;
    }

    private Object registrationLock(MetricKey key) {
        return registrationLocks[(key.hashCode() & Integer.MAX_VALUE) % registrationLocks.length];
    }

    private boolean registrationAllowed(FlexMetricTags metricTags) {
        EndpointScopeKey scopeKey = EndpointScopeKey.fromMetricTags(metricTags);
        if (scopeKey == null) {
            return true;
        }
        synchronized (endpointScopeLock) {
            EndpointScopeState state = endpointScopes.get(scopeKey);
            return state == null || state.references > 0 || !state.cleaned;
        }
    }

    private void releaseEndpointScope(EndpointScopeKey scopeKey) {
        synchronized (endpointScopeLock) {
            EndpointScopeState state = endpointScopes.get(scopeKey);
            if (state == null || state.references == 0) {
                return;
            }
            state.references--;
            if (state.references == 0) {
                state.retireAtMs = saturatedAdd(
                        System.currentTimeMillis(), state.retirementGraceMs);
            }
        }
    }

    private void runRetirementSweepSafely() {
        try {
            runRetirementSweep();
        } catch (Throwable t) {
            log.warn("Failed to retire endpoint metrics", t);
        }
    }

    void runRetirementSweep() {
        long now = System.currentTimeMillis();
        registrationLifecycleLock.writeLock().lock();
        try {
            synchronized (endpointScopeLock) {
                Set<EndpointScopeKey> scopesToClean = new HashSet<>();
                var iterator = endpointScopes.entrySet().iterator();
                while (iterator.hasNext()) {
                    Map.Entry<EndpointScopeKey, EndpointScopeState> entry = iterator.next();
                    EndpointScopeState state = entry.getValue();
                    if (state.references != 0) {
                        continue;
                    }
                    if (!state.cleaned && now >= state.retireAtMs) {
                        state.cleaned = true;
                        state.purgeAtMs = saturatedAdd(now, state.retirementGraceMs);
                        scopesToClean.add(entry.getKey());
                    } else if (state.cleaned && now >= state.purgeAtMs) {
                        iterator.remove();
                    }
                }
                removeEndpointMetrics(scopesToClean);
            }
        } finally {
            registrationLifecycleLock.writeLock().unlock();
        }
    }

    private void removeEndpointMetrics(Set<EndpointScopeKey> endpointScopesToClean) {
        if (endpointScopesToClean.isEmpty()) {
            return;
        }
        removeMatching(counterCache, endpointScopesToClean);
        removeMatching(timerCache, endpointScopesToClean);
        removeMatchingGauges(endpointScopesToClean);

        List<Meter> remainingMeters = new ArrayList<>(meterRegistry.getMeters());
        for (Meter meter : remainingMeters) {
            if (meter.getId().getName().startsWith(METRIC_PREFIX)
                    && endpointScopesToClean.contains(EndpointScopeKey.fromMeter(meter))) {
                meterRegistry.remove(meter);
            }
        }
        tagsCache.keySet().removeIf(tags -> endpointScopesToClean.contains(
                EndpointScopeKey.fromMetricTags(tags)));
    }

    private <T extends Meter> void removeMatching(ConcurrentHashMap<MetricKey, T> cache,
                                                   Set<EndpointScopeKey> endpointScopesToClean) {
        cache.entrySet().removeIf(entry -> {
            if (!endpointScopesToClean.contains(
                    EndpointScopeKey.fromMetricTags(entry.getKey().metricsTags()))) {
                return false;
            }
            meterRegistry.remove(entry.getValue());
            return true;
        });
    }

    private void removeMatchingGauges(Set<EndpointScopeKey> endpointScopesToClean) {
        gaugeCache.entrySet().removeIf(entry -> {
            if (!endpointScopesToClean.contains(
                    EndpointScopeKey.fromMetricTags(entry.getKey().metricsTags()))) {
                return false;
            }
            meterRegistry.remove(entry.getValue().gauge());
            return true;
        });
    }

    private static long saturatedAdd(long value, long increment) {
        if (increment > Long.MAX_VALUE - value) {
            return Long.MAX_VALUE;
        }
        return value + increment;
    }

    CacheStats cacheStats() {
        synchronized (endpointScopeLock) {
            return new CacheStats(gaugeCache.size(), counterCache.size(), timerCache.size(),
                    tagsCache.size(), endpointScopes.size());
        }
    }

    @PreDestroy
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            retirementExecutor.shutdownNow();
        }
    }

    /**
     * Convert FlexMetricTags to micrometer Tags.
     *
     * @param metricsTags FlexLB tags, may be null
     * @return micrometer Tags, never null
     */
    private Tags toMicrometerTags(FlexMetricTags metricsTags) {
        if (metricsTags == null || metricsTags.isEmpty()) {
            return Tags.empty();
        }
        Tags tags = Tags.empty();
        for (Map.Entry<String, String> entry : metricsTags.getTags().entrySet()) {
            tags = tags.and(entry.getKey(), entry.getValue());
        }
        return tags;
    }

    record CacheStats(int gauges, int counters, int timers, int tags, int endpointScopes) {
        int meters() {
            return gauges + counters + timers;
        }
    }

    private record MetricKey(String name, FlexMetricTags metricsTags) {
    }

    private record GaugeEntry(AtomicDouble value, Gauge gauge) {
    }

    private record EndpointScopeKey(String role, String engineIp, String engineIpPort) {
        private static EndpointScopeKey from(FlexMetricTags endpointTags) {
            Objects.requireNonNull(endpointTags, "endpointTags");
            Map<String, String> tags = endpointTags.getTags();
            String role = tags.get(ROLE_TAG);
            String engineIp = tags.get(ENGINE_IP_TAG);
            String engineIpPort = tags.get(ENGINE_IP_PORT_TAG);
            if (role == null || engineIp == null || engineIpPort == null) {
                throw new IllegalArgumentException(
                        "endpointTags must contain role, engineIp and engineIpPort");
            }
            return new EndpointScopeKey(role, engineIp, engineIpPort);
        }

        private static EndpointScopeKey fromMetricTags(FlexMetricTags metricTags) {
            if (metricTags == null) {
                return null;
            }
            Map<String, String> tags = metricTags.getTags();
            String role = tags.get(ROLE_TAG);
            String engineIp = tags.get(ENGINE_IP_TAG);
            String engineIpPort = tags.get(ENGINE_IP_PORT_TAG);
            return role == null || engineIp == null || engineIpPort == null
                    ? null
                    : new EndpointScopeKey(role, engineIp, engineIpPort);
        }

        private static EndpointScopeKey fromMeter(Meter meter) {
            String role = meter.getId().getTag(ROLE_TAG);
            String engineIp = meter.getId().getTag(ENGINE_IP_TAG);
            String engineIpPort = meter.getId().getTag(ENGINE_IP_PORT_TAG);
            return role == null || engineIp == null || engineIpPort == null
                    ? null
                    : new EndpointScopeKey(role, engineIp, engineIpPort);
        }
    }

    private static final class EndpointScopeState {
        private int references;
        private long retirementGraceMs;
        private long retireAtMs = Long.MAX_VALUE;
        private long purgeAtMs = Long.MAX_VALUE;
        private boolean cleaned;

    }

    @Override
    public String toString() {
        return "MicrometerFlexMonitor{meterRegistry=" + meterRegistry.getClass().getSimpleName() + "}";
    }
}
