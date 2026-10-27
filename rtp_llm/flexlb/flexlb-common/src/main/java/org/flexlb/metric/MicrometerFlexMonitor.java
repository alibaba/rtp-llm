package org.flexlb.metric;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Tags;
import io.micrometer.core.instrument.Timer;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

import java.util.Arrays;
import java.util.IdentityHashMap;
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
    // This monitor serializes only source registration/removal, never cached reports or scrapes.
    private final Map<Gauge, GaugeSources> gaugeSources = new IdentityHashMap<>();
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
            FlexMetricType metricType = metricTypes.getOrDefault(metricName, FlexMetricType.GAUGE);
            MetricKey key = new MetricKey(metricName, metricsTags);
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
                    gaugeEntry(key).value = value;
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
            FlexMetricType metricType = metricTypes.getOrDefault(metricName, FlexMetricType.GAUGE);
            MetricKey key = new MetricKey(metricName, metricsTags);
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
                        Counter.builder(METRIC_PREFIX + key.name())
                                .tags(resolveTags(key.metricsTags()))
                                .register(meterRegistry));
    }

    private Timer timer(MetricKey key) {
        return timerCache.computeIfAbsent(
                key,
                ignored ->
                        Timer.builder(METRIC_PREFIX + key.name())
                                .tags(resolveTags(key.metricsTags()))
                                .publishPercentileHistogram(true)
                                .publishPercentiles(0.5, 0.9, 0.95, 0.99)
                                .register(meterRegistry));
    }

    private GaugeEntry gaugeEntry(MetricKey key) {
        GaugeEntry cached = gaugeCache.get(key);
        if (cached != null) {
            return cached;
        }
        synchronized (gaugeSources) {
            cached = gaugeCache.get(key);
            if (cached != null) {
                return cached;
            }
            var created = new GaugeSources();
            Gauge gauge =
                    Gauge.builder(METRIC_PREFIX + key.name(), created, GaugeSources::read)
                            .tags(resolveTags(key.metricsTags()))
                            .register(meterRegistry);
            // Registration resolves all tag filters; the canonical group owns the gauge callback.
            GaugeSources sources = gaugeSources.get(gauge);
            if (sources == null) {
                sources = created;
                sources.meter = gauge;
                gaugeSources.put(gauge, sources);
            }
            var entry = new GaugeEntry(sources);
            sources.add(entry);
            gaugeCache.put(key, entry);
            return entry;
        }
    }

    @Override
    public void removeGauge(String metricName, FlexMetricTags metricsTags) {
        try {
            var key = new MetricKey(metricName, metricsTags);
            synchronized (gaugeSources) {
                GaugeEntry entry = gaugeCache.remove(key);
                if (entry == null) {
                    return;
                }
                GaugeSources sources = entry.sources;
                sources.remove(entry);
                if (key.metricsTags() != null) {
                    tagsCache.remove(key.metricsTags());
                }
                if (sources.entries == null) {
                    // Retain an empty group if backend removal fails, so a later report can reuse
                    // it.
                    meterRegistry.remove(sources.meter);
                    gaugeSources.remove(sources.meter);
                }
            }
        } catch (Throwable failure) {
            SafeFlexMonitor.logFailure(metricName, failure);
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

    private record MetricKey(String name, FlexMetricTags metricsTags) {
        private MetricKey {
            if (metricsTags != null && metricsTags.isEmpty()) {
                metricsTags = null;
            }
        }
    }

    private static final class GaugeEntry {
        private volatile double value;
        private final GaugeSources sources;

        private GaugeEntry(GaugeSources sources) {
            this.sources = sources;
        }
    }

    private static final class GaugeSources {
        private Gauge meter;
        // Null, one entry, or an immutable array of two or more entries.
        private volatile Object entries;

        // Writers hold gaugeSources. Readers capture one published membership value.
        private void add(GaugeEntry entry) {
            if (entries instanceof GaugeEntry[] current) {
                var next = Arrays.copyOf(current, current.length + 1);
                next[next.length - 1] = entry;
                entries = next;
            } else {
                entries = entries == null ? entry : new GaugeEntry[] {(GaugeEntry) entries, entry};
            }
        }

        private void remove(GaugeEntry entry) {
            if (entries == entry) {
                entries = null;
                return;
            }
            var current = (GaugeEntry[]) entries;
            for (int i = 0; i < current.length; i++) {
                if (current[i] == entry) {
                    var next = new GaugeEntry[current.length - 1];
                    System.arraycopy(current, 0, next, 0, i);
                    System.arraycopy(current, i + 1, next, i, next.length - i);
                    entries = next.length == 1 ? next[0] : next;
                    return;
                }
            }
        }

        private double read() {
            Object current = entries;
            if (current instanceof GaugeEntry entry) return entry.value;
            double total = 0.0;
            if (current instanceof GaugeEntry[] group) {
                for (GaugeEntry entry : group) total += entry.value;
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
