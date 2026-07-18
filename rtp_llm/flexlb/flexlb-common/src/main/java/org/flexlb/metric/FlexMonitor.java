package org.flexlb.metric;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

/**
 * FlexMonitor - Unified monitoring interface
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
     * Acquire ownership of all metrics carrying the supplied endpoint identity tags.
     * The last closed lease allows the implementation to retire those metrics after
     * {@code retirementGraceMs}. Implementations without dynamic metric storage may
     * keep the default no-op lease.
     *
     * @param endpointTags endpoint identity tags (role, engineIp, engineIpPort)
     * @param retirementGraceMs time allowed for in-flight callbacks to finish
     * @return an idempotent ownership lease
     */
    default MetricLease acquireEndpointScope(FlexMetricTags endpointTags, long retirementGraceMs) {
        return MetricLease.noop();
    }

    @Override
    default void close() {
        // No-op by default.
    }
}
