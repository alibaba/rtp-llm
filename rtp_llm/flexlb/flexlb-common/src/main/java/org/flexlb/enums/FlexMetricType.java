package org.flexlb.enums;

/**
 * FlexMetricType - Monitoring metric type enumeration
 *
 * @author saichen.sm
 */
public enum FlexMetricType {

    /**
     * GAUGE type - Instantaneous value metric
     * Represents instantaneous value at a specific point in time, such as CPU usage, memory usage, etc.
     * Characteristics: Value can change arbitrarily (increase or decrease)
     */
    GAUGE("GAUGE", "Instantaneous value metric, representing immediate value at a specific point in time"),
    /**
     * COUNTER type - Cumulative counter
     * Represents cumulative count value, can only increase, such as total requests, total errors, etc.
     * Characteristics: Value can only increase, not decrease (unless reset)
     */
    COUNTER("COUNTER", "Cumulative counter, value can only increase"),
    /**
     * QPS type - Queries Per Second
     * Represents request processing rate per second, typically used to measure system throughput.
     * Characteristics: Rate value calculated based on time window
     */
    QPS("QPS", "Queries Per Second, measures system request processing speed"),
    /**
     * TIMER type - Duration distribution metric
     * Records duration distributions (p50/p90/p95/p99) via Micrometer Timer.
     * Characteristics: Provides histogram data (count, sum, percentiles) instead of instantaneous value
     */
    TIMER("TIMER", "Timer metric for recording duration distributions (p50/p90/p99)");
    private final String type;
    private final String description;

    /**
     * Constructor
     *
     * @param type        Metric type name
     * @param description Metric type description
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
