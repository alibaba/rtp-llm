package org.flexlb.metric;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

/**
 * FlexMonitor - Unified monitoring interface
 *
 * <p>Operations are best-effort: backend failures must not escape to business callers.
 * Use {@link SafeFlexMonitor#wrap(FlexMonitor)} for external implementations.
 *
 * @author saichen.sm
 */
public interface FlexMonitor extends AutoCloseable {

    /**
     * Register monitoring metric
     *
     * @param metricName Metric name
     * @param metricType Metric type
     */
    void register(String metricName, FlexMetricType metricType);

    /**
     * Register monitoring metric
     *
     * @param metricName   Metric name
     * @param metricType   Metric type
     * @param priorityType Priority type
     */
    void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType);

    /**
     * Register monitoring metric
     *
     * @param metricName     Metric name
     * @param metricType     Metric type
     * @param statisticsType Statistics type
     */
    void register(String metricName, FlexMetricType metricType, int statisticsType);

    /**
     * Report monitoring data
     *
     * @param metricName Metric name
     * @param value      Metric value
     */
    void report(String metricName, double value);

    /**
     * Report monitoring data
     *
     * @param metricName  Metric name
     * @param metricsTags Tags object
     * @param value       Metric value
     */
    void report(String metricName, FlexMetricTags metricsTags, double value);

    /**
     * Prepare a tagged metric before it enters a hot reporting path.
     * Implementations that do not require registration may keep the default no-op.
     *
     * @param metricName Metric name
     * @param metricsTags Tags object
     */
    default void prepare(String metricName, FlexMetricTags metricsTags) {
        // No-op by default.
    }

    /**
     * Withdraw one gauge source after its owner stops reporting. A later report registers it anew.
     * Backends without source removal at least clear its contribution.
     */
    default void removeGauge(String metricName, FlexMetricTags metricsTags) {
        report(metricName, metricsTags, 0.0);
    }

    @Override
    default void close() {
        // No-op by default.
    }
}
