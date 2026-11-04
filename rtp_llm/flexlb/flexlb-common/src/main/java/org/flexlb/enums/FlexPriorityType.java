package org.flexlb.enums;

/**
 * FlexPriorityType - Monitoring metric priority type enumeration
 *
 * @author saichen.sm
 */
public enum FlexPriorityType {

    /**
     * Different sampling precision levels
     */
    PRECISE(1),
    CRITICAL(5),
    MAJOR(10),
    NORMAL(20),
    TRIVIAL(60);

    private final int windowSeconds;

    FlexPriorityType(int windowSeconds) {
        this.windowSeconds = windowSeconds;
    }

    /**
     * Returns the aggregation window used by rate metrics.
     *
     * @return aggregation window in seconds
     */
    public int getWindowSeconds() {
        return windowSeconds;
    }
}
