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

    private final MeterRegistry meterRegistry;
    private final ConcurrentHashMap<String, FlexMetricType> metricTypes = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, AtomicDouble> gaugeValues = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Counter> counterCache = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Timer> timerCache = new ConcurrentHashMap<>();
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
        String prefixedName = METRIC_PREFIX + metricName;
        FlexMetricType metricType = metricTypes.getOrDefault(metricName, FlexMetricType.GAUGE);
        Tags tags = (metricsTags == null || metricsTags.isEmpty())
                ? Tags.empty()
                // toMicrometerTags is a lightweight pure function with no external lock,
                // so computeIfAbsent is safe here — no lock nesting risk.
                : tagsCache.computeIfAbsent(metricsTags, this::toMicrometerTags);

        try {
            switch (metricType) {
                case QPS:
                case COUNTER:
                    String counterKey = prefixedName + "|" + tags;
                    // Avoid computeIfAbsent: it holds CHM bin lock while calling register()
                    // which acquires Micrometer's global meterMapLock, causing lock nesting.
                    Counter counter = counterCache.get(counterKey);
                    if (counter == null) {
                        counter = Counter.builder(prefixedName).tags(tags).register(meterRegistry);
                        Counter existing = counterCache.putIfAbsent(counterKey, counter);
                        if (existing != null) {
                            counter = existing;
                        }
                    }
                    counter.increment(value);
                    break;
                case TIMER:
                    String timerKey = prefixedName + "|" + tags;
                    // Avoid computeIfAbsent: it holds CHM bin lock while calling register()
                    // which acquires Micrometer's global meterMapLock, causing lock nesting.
                    Timer timer = timerCache.get(timerKey);
                    if (timer == null) {
                        timer = Timer.builder(prefixedName)
                                .tags(tags)
                                .publishPercentileHistogram(true)
                                .publishPercentiles(0.5, 0.9, 0.95, 0.99)
                                .register(meterRegistry);
                        Timer existing = timerCache.putIfAbsent(timerKey, timer);
                        if (existing != null) {
                            timer = existing;
                        }
                    }
                    timer.record((long) value, TimeUnit.MILLISECONDS);
                    break;
                case GAUGE:
                default:
                    reportGauge(prefixedName, tags, value);
                    break;
            }
        } catch (Exception e) {
            log.warn("Failed to report metric {}: {}", metricName, e.getMessage());
        }
    }

    /**
     * Report a gauge value. Uses an AtomicDouble per metric-name+tags combination
     * as the gauge reference object, so each unique tag set gets its own gauge.
     *
     * @param name  prefixed metric name
     * @param tags  micrometer tags
     * @param value absolute gauge value
     */
    private void reportGauge(String name, Tags tags, double value) {
        String gaugeKey = name + "|" + tags;
        // Avoid computeIfAbsent: it holds CHM bin lock while calling register()
        // which acquires Micrometer's global meterMapLock, causing lock nesting.
        AtomicDouble atomicDouble = gaugeValues.get(gaugeKey);
        if (atomicDouble == null) {
            atomicDouble = new AtomicDouble(0.0);
            Gauge.builder(name, atomicDouble, AtomicDouble::get)
                    .tags(tags)
                    .register(meterRegistry);
            AtomicDouble existing = gaugeValues.putIfAbsent(gaugeKey, atomicDouble);
            if (existing != null) {
                atomicDouble = existing;
            }
        }
        atomicDouble.set(value);
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

    @Override
    public String toString() {
        return "MicrometerFlexMonitor{meterRegistry=" + meterRegistry.getClass().getSimpleName() + "}";
    }
}
