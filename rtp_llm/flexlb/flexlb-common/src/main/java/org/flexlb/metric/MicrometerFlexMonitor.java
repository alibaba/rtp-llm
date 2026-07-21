package org.flexlb.metric;

import com.google.common.util.concurrent.AtomicDouble;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

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

    /**
     * Constructor.
     *
     * @param meterRegistry micrometer MeterRegistry (typically PrometheusMeterRegistry)
     */
    public MicrometerFlexMonitor(MeterRegistry meterRegistry) {
        this.meterRegistry = meterRegistry;
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

    private Counter counter(MetricKey key) {
        return counterCache.computeIfAbsent(key, ignored -> Counter.builder(key.name())
                .tags(resolveTags(key.metricsTags()))
                .register(meterRegistry));
    }

    private Timer timer(MetricKey key) {
        return timerCache.computeIfAbsent(key, ignored -> Timer.builder(key.name())
                .tags(resolveTags(key.metricsTags()))
                .publishPercentileHistogram(true)
                .publishPercentiles(0.5, 0.9, 0.95, 0.99)
                .register(meterRegistry));
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
        return gaugeCache.computeIfAbsent(key, ignored -> {
            AtomicDouble value = new AtomicDouble(0.0);
            Gauge gauge = Gauge.builder(key.name(), value, AtomicDouble::get)
                    .tags(resolveTags(key.metricsTags()))
                    .register(meterRegistry);
            return new GaugeEntry(value, gauge);
        });
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

    private record MetricKey(String name, FlexMetricTags metricsTags) {
    }

    private record GaugeEntry(AtomicDouble value, Gauge gauge) {
    }

    @Override
    public String toString() {
        return "MicrometerFlexMonitor{meterRegistry=" + meterRegistry.getClass().getSimpleName() + "}";
    }
}
