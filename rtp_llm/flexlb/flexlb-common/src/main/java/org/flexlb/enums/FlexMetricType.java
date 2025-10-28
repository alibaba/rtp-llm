package org.flexlb.enums;

/**
 * FlexMetricType - 监控指标类型枚举
 *
 * @author saichen.sm
 */
public enum FlexMetricType {

    /**
     * GAUGE类型 - 瞬时值指标
     * 表示某个时间点的瞬时值，如CPU使用率、内存使用量等。
     * 特点：值可以任意变化（增加或减少）
     */
    GAUGE("GAUGE", "瞬时值指标，表示某个时间点的即时数值"),
    /**
     * COUNTER类型 - 累积计数器
     * 表示累积的计数值，只能递增，如请求总数、错误总数等。
     * 特点：值只能增加，不能减少（除非重置）
     */
    COUNTER("COUNTER", "累积计数器，值只能递增"),
    /**
     * QPS类型 - 每秒查询率
     * 表示每秒的请求处理速率，通常用于衡量系统吞吐量。
     * 特点：基于时间窗口计算的速率值
     */
    QPS("QPS", "每秒查询率，衡量系统处理请求的速度");
    private final String type;
    private final String description;

    /**
     * 构造函数
     *
     * @param type        指标类型名称
     * @param description 指标类型描述
     */
    FlexMetricType(String type, String description) {
        this.type = type;
        this.description = description;
    }

    @Override
    public String toString() {
        return "FlexMetricType{" +
                "type='" + type + '\'' +
                ", description='" + description + '\'' +
                '}';
    }
}
