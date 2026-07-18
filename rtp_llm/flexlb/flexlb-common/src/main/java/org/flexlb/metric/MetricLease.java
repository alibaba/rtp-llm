package org.flexlb.metric;

/**
 * Idempotent ownership handle for metrics tied to a dynamic resource.
 */
@FunctionalInterface
public interface MetricLease extends AutoCloseable {

    MetricLease NOOP = () -> { };

    static MetricLease noop() {
        return NOOP;
    }

    @Override
    void close();
}
