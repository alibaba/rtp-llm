package org.flexlb.monitor.prometheus;

import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.Counter;
import io.prometheus.client.Gauge;
import io.prometheus.client.Summary;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.FlexStatisticsType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * {@link FlexMonitor} implementation backed by a Prometheus {@link CollectorRegistry}.
 *
 * <p>Callers must register a metric with its complete tag-key schema before reporting values. Registration
 * normalizes the metric name with a {@code flexlb_} prefix, sorts the tag keys, and creates the corresponding
 * Prometheus collector exactly once. Statistics registrations create a {@link Summary}; non-statistics
 * {@link FlexMetricType#GAUGE gauges} create a {@link Gauge}; {@link FlexMetricType#QPS QPS} metrics create a
 * windowed {@link Gauge}; and counters create a {@link Counter}.
 *
 * <p>A report is accepted only when the metric has been registered and its tag keys exactly match the schema
 * captured at registration. Tag values are then supplied in the same sorted-key order used to create the
 * collector. Reports for unregistered metrics, changed tag schemas, disabled collectors, or SDK failures are
 * dropped and logged instead of changing collector definitions at runtime.
 *
 * <p>This fixed-schema model follows Prometheus exposition requirements: each exposed sample must have a unique
 * combination of metric name and labels. Binding creation is therefore performed during registration. The
 * immutable binding can then be used directly from {@link #report(String, FlexMetricTags, double)} without a
 * report-path initialization lock; the concurrent metric map provides one-time registration per metric name.
 *
 * <p>See <a href="https://prometheus.io/docs/instrumenting/exposition_formats/">Prometheus exposition
 * formats</a> for the metric and label constraints.
 */
public class PrometheusFlexMonitor implements FlexMonitor, AutoCloseable {

    private static final Logger log = LoggerFactory.getLogger(PrometheusFlexMonitor.class);
    private static final FlexPriorityType DEFAULT_QPS_PRIORITY = FlexPriorityType.NORMAL;
    private static final List<QuantileConfiguration> QUANTILE_CONFIGURATIONS = List.of(
            new QuantileConfiguration(FlexStatisticsType.MIN, 0.0, 0.0),
            new QuantileConfiguration(FlexStatisticsType.MAX, 1.0, 0.0),
            new QuantileConfiguration(FlexStatisticsType.PERCENTILE_75, 0.75, 0.01),
            new QuantileConfiguration(FlexStatisticsType.PERCENTILE_95, 0.95, 0.005),
            new QuantileConfiguration(FlexStatisticsType.PERCENTILE_99, 0.99, 0.001));

    private final CollectorRegistry registry;
    private final Map<String, MetricState> metrics = new ConcurrentHashMap<>();
    private final AtomicLong qpsSnapshotTicks = new AtomicLong();
    private final ScheduledExecutorService qpsSnapshotExecutor;

    /**
     * Creates a monitor that registers collectors in {@code registry}.
     *
     * @param registry Prometheus collector registry that owns all collectors created by this monitor
     */
    public PrometheusFlexMonitor(CollectorRegistry registry) {
        this.registry = Objects.requireNonNull(registry, "registry");
        qpsSnapshotExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "flexlb-prometheus-qps-snapshot");
            thread.setDaemon(true);
            return thread;
        });
        qpsSnapshotExecutor.scheduleAtFixedRate(this::snapshotQpsMetrics, 1, 1, TimeUnit.SECONDS);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: registers the metric with an empty label schema.
     */
    @Override
    public void register(String metricName, FlexMetricType metricType) {
        register(metricName, metricType, FlexMetricTags.of());
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: creates the collector during registration and freezes the sorted tag keys
     * as its label schema.
     */
    @Override
    public void register(String metricName, FlexMetricType metricType, FlexMetricTags metricsTags) {
        register(metricName, metricType, DEFAULT_QPS_PRIORITY, 0, false, metricsTags);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: {@link FlexPriorityType} defines the aggregation window for
     * {@link FlexMetricType#QPS QPS} metrics and is ignored by all other metric types.
     */
    @Override
    public void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType) {
        register(metricName, metricType, priorityType, FlexMetricTags.of());
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: {@code priorityType} defines the QPS aggregation window while
     * {@code metricsTags} defines the immutable collector label schema.
     */
    @Override
    public void register(
            String metricName,
            FlexMetricType metricType,
            FlexPriorityType priorityType,
            FlexMetricTags metricsTags) {
        register(metricName, metricType, priorityType, 0, false, metricsTags);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: creates a {@link Summary} with an empty label schema.
     */
    @Override
    public void register(String metricName, FlexMetricType metricType, int statisticsType) {
        register(metricName, metricType, statisticsType, FlexMetricTags.of());
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: creates a {@link Summary}, enabling the quantiles selected by
     * {@code statisticsType}, and freezes {@code metricsTags} as its label schema.
     */
    @Override
    public void register(
            String metricName,
            FlexMetricType metricType,
            int statisticsType,
            FlexMetricTags metricsTags) {
        register(metricName, metricType, DEFAULT_QPS_PRIORITY, statisticsType, true, metricsTags);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: reports against the empty label schema.
     */
    @Override
    public void report(String metricName, double value) {
        report(metricName, FlexMetricTags.of(), value);
    }

    /**
     * {@inheritDoc}
     *
     * <p>Prometheus-specific behavior: accepts reports only for an existing collector whose sorted tag keys match
     * the registration schema. The report path only updates the immutable binding and never creates a collector.
     */
    @Override
    public void report(String metricName, FlexMetricTags metricsTags, double value) {
        MetricState state = metrics.get(metricName);
        if (state == null) {
            log.warn("Dropping unregistered Prometheus metric: {}", metricName);
            return;
        }

        List<Map.Entry<String, String>> tags = sortedTags(metricsTags);
        List<String> labelNames = tags.stream().map(Map.Entry::getKey).toList();
        if (!state.labelNames.equals(labelNames)) {
            log.warn("Dropping Prometheus metric report with changed label schema: {}", metricName);
            return;
        }
        if (state.binding == null) {
            return;
        }
        try {
            state.binding.update(tags.stream().map(Map.Entry::getValue).toArray(String[]::new), value);
        } catch (RuntimeException error) {
            log.warn("Dropping Prometheus metric report after SDK failure: {}: {}", metricName, error.getMessage());
        }
    }

    /**
     * Atomically creates the metric state for the first registration of {@code metricName}.
     *
     * <p>Later registrations intentionally retain the original collector and label schema.
     */
    private void register(
            String metricName,
            FlexMetricType metricType,
            FlexPriorityType priorityType,
            int statistics,
            boolean statisticsEnabled,
            FlexMetricTags metricsTags) {
        List<String> labelNames = sortedTags(metricsTags).stream().map(Map.Entry::getKey).toList();
        metrics.computeIfAbsent(metricName, ignored -> createMetricState(
                metricName, metricType, priorityType, statistics, statisticsEnabled, labelNames));
    }

    /**
     * Creates immutable registration state, preserving the label schema even if the Prometheus SDK rejects the
     * collector definition. Such a state disables later reports without repeatedly attempting registration.
     */
    private MetricState createMetricState(
            String metricName,
            FlexMetricType metricType,
            FlexPriorityType priorityType,
            int statistics,
            boolean statisticsEnabled,
            List<String> labelNames) {
        try {
            Binding binding = createBinding(
                    metricName, labelNames, metricType, priorityType, statistics, statisticsEnabled);
            return new MetricState(labelNames, binding);
        } catch (RuntimeException error) {
            log.warn("Disabling Prometheus metric after registration failure: {}: {}", metricName, error.getMessage());
            return new MetricState(labelNames, null);
        }
    }

    /**
     * Creates the Prometheus collector and returns its pre-bound update operation.
     *
     * <p>Statistics registrations use a summary; otherwise gauges use {@link Gauge}, QPS uses a windowed gauge,
     * and counters use {@link Counter}.
     */
    private Binding createBinding(
            String metricName,
            List<String> labelNames,
            FlexMetricType metricType,
            FlexPriorityType priorityType,
            int statistics,
            boolean statisticsEnabled) {
        String[] labels = labelNames.toArray(String[]::new);
        String name = normalize(metricName);
        String help = "FlexLB metric " + metricName;
        if (statisticsEnabled) {
            Summary.Builder builder = Summary.build().name(name).help(help).labelNames(labels);
            addQuantiles(builder, statistics);
            Summary summary = builder.register(registry);
            return (values, value) -> summary.labels(values).observe(value);
        }
        if (metricType == FlexMetricType.GAUGE) {
            Gauge gauge = Gauge.build().name(name).help(help).labelNames(labels).register(registry);
            return (values, value) -> gauge.labels(values).set(value);
        }
        if (metricType == FlexMetricType.QPS) {
            Gauge gauge = Gauge.build().name(name).help(help).labelNames(labels).register(registry);
            return new QpsBinding(gauge, priorityType.getWindowSeconds(), labels.length == 0);
        }
        Counter counter = Counter.build().name(name).help(help).labelNames(labels).register(registry);
        return (values, value) -> counter.labels(values).inc(value);
    }

    private void snapshotQpsMetrics() {
        long tick = qpsSnapshotTicks.incrementAndGet();
        try {
            metrics.values().forEach(state -> {
                if (state.binding != null) {
                    state.binding.snapshot(tick);
                }
            });
        } catch (RuntimeException error) {
            log.warn("Unable to publish Prometheus QPS snapshot: {}", error.getMessage());
        }
    }

    /**
     * Stops the QPS snapshot thread when the monitor is destroyed by Spring.
     */
    @Override
    public void close() {
        qpsSnapshotExecutor.shutdownNow();
    }

    /**
     * Adds each supported quantile selected by the FlexLB statistics bitmask.
     */
    private static void addQuantiles(Summary.Builder builder, int statistics) {
        for (QuantileConfiguration quantile : QUANTILE_CONFIGURATIONS) {
            if ((statistics & quantile.statisticsFlag) != 0) {
                builder.quantile(quantile.quantile, quantile.error);
            }
        }
    }

    private static String normalize(String metricName) {
        String normalized = metricName.replaceAll("[^a-zA-Z0-9_]", "_")
                .replaceAll("_+", "_");
        while (normalized.startsWith("flexlb_")) {
            normalized = normalized.substring("flexlb_".length());
        }
        return "flexlb_" + normalized;
    }

    /**
     * Returns tags in the canonical label order shared by registration and reporting.
     */
    private static List<Map.Entry<String, String>> sortedTags(FlexMetricTags metricsTags) {
        return metricsTags.getTags().entrySet()
                .stream()
                .sorted(Map.Entry.comparingByKey())
                .toList();
    }

    private record  MetricState(List<String> labelNames, Binding binding) {

        private MetricState(List<String> labelNames, Binding binding) {
            this.labelNames = List.copyOf(labelNames);
            this.binding = binding;
        }
    }

    private record QuantileConfiguration(int statisticsFlag, double quantile, double error) {
    }

    @FunctionalInterface
    private interface Binding {

        /**
         * Updates a collector that was fully bound during metric registration.
         *
         * <p>{@code labelValues} must have the same cardinality and sorted-key order as the registered label schema.
         * Implementations apply the metric-type-specific Prometheus operation: set for gauges, accumulate for QPS,
         * increment for counters, or observe for summaries.
         *
         * @param labelValues ordered label values for one Prometheus sample
         * @param value metric value to apply
         */
        void update(String[] labelValues, double value);

        /**
         * Publishes an eligible QPS window. Non-QPS bindings do nothing.
         */
        default void snapshot(long tick) {
        }
    }

    /**
     * Holds one QPS accumulator per label-value series and publishes a KMonitor-compatible window average.
     */
    private static final class QpsBinding implements Binding {

        private final Gauge gauge;
        private final int windowSeconds;
        private final Map<List<String>, DoubleAdder> valuesByLabelSeries = new ConcurrentHashMap<>();

        private QpsBinding(Gauge gauge, int windowSeconds, boolean unlabelled) {
            this.gauge = gauge;
            this.windowSeconds = windowSeconds;
            if (unlabelled) {
                gauge.labels().set(0.0);
            }
        }

        @Override
        public void update(String[] labelValues, double value) {
            List<String> labelSeries = List.copyOf(Arrays.asList(labelValues.clone()));
            valuesByLabelSeries.computeIfAbsent(labelSeries, ignored -> new DoubleAdder()).add(value);
        }

        @Override
        public void snapshot(long tick) {
            if (tick % windowSeconds != 0) {
                return;
            }
            valuesByLabelSeries.forEach((labelSeries, value) -> gauge.labels(labelSeries.toArray(String[]::new))
                    .set(value.sumThenReset() / windowSeconds));
        }
    }

}
