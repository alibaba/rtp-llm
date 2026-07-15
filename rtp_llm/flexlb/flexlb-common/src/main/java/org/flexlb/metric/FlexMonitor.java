package org.flexlb.metric;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

/**
 * FlexMonitor - Unified monitoring interface
 *
 * @author saichen.sm
 */
public interface FlexMonitor {

    /**
     * Register monitoring metric
     *
     * @param metricName Metric name
     * @param metricType Metric type
     */
    void register(String metricName, FlexMetricType metricType);

    /**
     * Register monitoring metric with its tag schema.
     *
     * @param metricName  Metric name
     * @param metricType  Metric type
     * @param metricsTags Tags object containing the metric's tag keys
     */
    default void register(String metricName, FlexMetricType metricType, FlexMetricTags metricsTags) {
        register(metricName, metricType);
    }

    /**
     * Register monitoring metric
     *
     * @param metricName   Metric name
     * @param metricType   Metric type
     * @param priorityType Priority type
     */
    void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType);

    /**
     * Register monitoring metric with priority and its tag schema.
     *
     * @param metricName   Metric name
     * @param metricType   Metric type
     * @param priorityType Priority type
     * @param metricsTags  Tags object containing the metric's tag keys
     */
    default void register(
            String metricName,
            FlexMetricType metricType,
            FlexPriorityType priorityType,
            FlexMetricTags metricsTags) {
        register(metricName, metricType, priorityType);
    }

    /**
     * Register monitoring metric
     *
     * @param metricName     Metric name
     * @param metricType     Metric type
     * @param statisticsType Statistics type
     */
    void register(String metricName, FlexMetricType metricType, int statisticsType);

    /**
     * Register monitoring metric with statistics and its tag schema.
     *
     * @param metricName     Metric name
     * @param metricType     Metric type
     * @param statisticsType Statistics type
     * @param metricsTags    Tags object containing the metric's tag keys
     */
    default void register(
            String metricName,
            FlexMetricType metricType,
            int statisticsType,
            FlexMetricTags metricsTags) {
        register(metricName, metricType, statisticsType);
    }

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
}
