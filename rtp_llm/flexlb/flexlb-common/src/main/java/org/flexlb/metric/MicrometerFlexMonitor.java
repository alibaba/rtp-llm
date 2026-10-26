package org.flexlb.metric;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

import java.util.Arrays;
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
 *   <li>{@link FlexMetricType#GAUGE} → micrometer Gauge (sum of latest source values)</li>
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
     * Set by {@link org.flexlb.config.WhitelistMetricsFilterConfig} from the
     * {@code flexlb.monitor.metric-whitelist} property.
     */
    private static volatile Set<String> ALLOWED_METRICS = null;

    /**
     * Sets the allowlist of metric names (without the {@code flexlb.} prefix).
     * Pass {@code null} to disable filtering (allow all metrics).
     */
    public static void setAllowedMetrics(Set<String> metrics) {
        ALLOWED_METRICS = metrics;
        try {
            log.info(
                    "MicrometerFlexMonitor allowlist set: {} metrics",
                    metrics == null ? "unlimited" : metrics.size());
        } catch (Throwable failure) {
            SafeFlexMonitor.logFailure("allowlist", failure);
        }
    }

    private final MeterRegistry meterRegistry;
    private final ConcurrentHashMap<String, FlexMetricType> metricTypes = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<MetricKey, GaugeEntry> gaugeCache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Meter.Id, GaugeSources> gaugeSources =
            new ConcurrentHashMap<>();
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
    }

    @Override
    public void register(String metricName, FlexMetricType metricType) {
        try {
            Set<String> allowedMetrics = ALLOWED_METRICS;
            if (allowedMetrics != null && !allowedMetrics.contains(metricName)) {
                return;
            }
            metricTypes.put(metricName, metricType);
        } catch (Throwable failure) {
            SafeFlexMonitor.logFailure(metricName, failure);
        }
    }

    @Override
    public void register(
            String metricName, FlexMetricType metricType, FlexPriorityType priorityType) {
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
        try {
            Set<String> allowedMetrics = ALLOWED_METRICS;
            if (allowedMetrics != null && !allowedMetrics.contains(metricName)) {
                return;
            }
            String prefixedName = METRIC_PREFIX + metricName;
            FlexMetricType metricType = metricTypes.getOrDefault(metricName, FlexMetricType.GAUGE);
            MetricKey key = new MetricKey(prefixedName, metricsTags);
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
        } catch (Throwable failure) {
            SafeFlexMonitor.logFailure(metricName, failure);
        }
    }

    @Override
    public void prepare(String metricName, FlexMetricTags metricsTags) {
        try {
            Set<String> allowedMetrics = ALLOWED_METRICS;
            if (allowedMetrics != null && !allowedMetrics.contains(metricName)) {
                return;
            }
            String prefixedName = METRIC_PREFIX + metricName;
            FlexMetricType metricType = metricTypes.getOrDefault(metricName, FlexMetricType.GAUGE);
            MetricKey key = new MetricKey(prefixedName, metricsTags);
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
        } catch (Throwable failure) {
            SafeFlexMonitor.logFailure(metricName, failure);
        }
    }

    private Tags resolveTags(FlexMetricTags metricsTags) {
        return (metricsTags == null || metricsTags.isEmpty())
                ? Tags.empty()
                : tagsCache.computeIfAbsent(metricsTags, this::toMicrometerTags);
    }

    private Counter counter(MetricKey key) {
        return counterCache.computeIfAbsent(
                key,
                ignored ->
                        Counter.builder(key.name())
                                .tags(resolveTags(key.metricsTags()))
                                .register(meterRegistry));
    }

    private Timer timer(MetricKey key) {
        return timerCache.computeIfAbsent(
                key,
                ignored ->
                        Timer.builder(key.name())
                                .tags(resolveTags(key.metricsTags()))
                                .publishPercentileHistogram(true)
                                .publishPercentiles(0.5, 0.9, 0.95, 0.99)
                                .register(meterRegistry));
    }

    /**
     * Replace this source's latest value. Sources collapsed by registry filters are summed on scrape.
     *
     * @param value absolute gauge value
     */
    private void reportGauge(MetricKey key, double value) {
        GaugeEntry entry = gaugeEntry(key);
        if (entry != null) {
            entry.value = value;
        }
    }

    private GaugeEntry gaugeEntry(MetricKey key) {
        GaugeEntry cached = gaugeCache.get(key);
        if (cached != null) {
            return cached;
        }
        return gaugeCache.computeIfAbsent(
                key,
                ignored -> {
                    GaugeEntry entry = new GaugeEntry();
                    Gauge gauge =
                            Gauge.builder(key.name(), entry, GaugeEntry::read)
                                    .tags(resolveTags(key.metricsTags()))
                                    .register(meterRegistry);
                    // Registration returns the canonical meter after all tag filters have run.
                    entry.sources =
                            gaugeSources.computeIfAbsent(
                                    gauge.getId(), unused -> new GaugeSources());
                    entry.sources.add(entry);
                    return entry;
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
        private MetricKey {
            if (metricsTags != null && metricsTags.isEmpty()) {
                metricsTags = null;
            }
        }
    }

    private static final class GaugeEntry {
        private volatile double value;
        private volatile GaugeSources sources;

        private double read() {
            var current = sources;
            return current == null ? 0.0 : current.read();
        }
    }

    private static final class GaugeSources {
        private GaugeEntry first;
        private volatile GaugeEntry[] entries;
        private volatile int size;

        private synchronized void add(GaugeEntry entry) {
            int count = size;
            if (count == 0) {
                first = entry;
            } else if (count == 1) {
                entries = new GaugeEntry[] {first, entry};
            } else {
                if (count == entries.length) {
                    entries = Arrays.copyOf(entries, entries.length * 2);
                }
                entries[count] = entry;
            }
            // Publish the array and its populated prefix to readers without locking
            // updates/scrapes.
            size = count + 1;
        }

        private double read() {
            int count = size;
            if (count <= 1) {
                return count == 0 ? 0.0 : first.value;
            }
            // A later array can only contain a longer initialized prefix than the captured size.
            GaugeEntry[] current = entries;
            double total = 0.0;
            for (int i = 0; i < count; i++) {
                total += current[i].value;
            }
            return total;
        }
    }

    @Override
    public String toString() {
        return "MicrometerFlexMonitor{meterRegistry="
                + (meterRegistry == null ? "disabled" : meterRegistry.getClass().getSimpleName())
                + "}";
    }
}
