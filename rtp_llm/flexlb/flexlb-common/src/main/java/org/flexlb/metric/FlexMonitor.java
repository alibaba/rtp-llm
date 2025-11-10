package org.flexlb.metric;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

/**
 * FlexMonitor - 统一监控接口
 *
 * @author saichen.sm
 */
public interface FlexMonitor {

    /**
     * 注册监控指标
     *
     * @param metricName 指标名称
     * @param metricType 指标类型
     */
    void register(String metricName, FlexMetricType metricType);

    /**
     * 注册监控指标
     *
     * @param metricName   指标名称
     * @param metricType   指标类型
     * @param priorityType 优先级类型
     */
    void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType);

    /**
     * 注册监控指标
     *
     * @param metricName     指标名称
     * @param metricType     指标类型
     * @param statisticsType 统计类型
     */
    void register(String metricName, FlexMetricType metricType, int statisticsType);

    /**
     * 上报监控数据
     *
     * @param metricName 指标名称
     * @param value      指标值
     */
    void report(String metricName, double value);

    /**
     * 上报监控数据
     *
     * @param metricName  指标名称
     * @param metricsTags 标签对象
     * @param value       指标值
     */
    void report(String metricName, FlexMetricTags metricsTags, double value);
}
