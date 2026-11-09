package org.flexlb.metric;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

/**
 * Best-effort boundary for monitoring backends.
 */
@Slf4j
public final class SafeFlexMonitor implements FlexMonitor {
    private final FlexMonitor delegate;
    private final FlexMonitor observations;

    private SafeFlexMonitor(FlexMonitor delegate) {
        this(delegate, delegate);
    }

    private SafeFlexMonitor(FlexMonitor delegate, FlexMonitor observations) {
        this.delegate = delegate;
        this.observations = observations;
    }

    public static FlexMonitor wrap(FlexMonitor monitor) {
        if (monitor == null) {
            return NoOpFlexMonitor.getInstance();
        }
        if (monitor instanceof SafeFlexMonitor || monitor == NoOpFlexMonitor.getInstance()) {
            return monitor;
        }
        return new SafeFlexMonitor(monitor);
    }

    /** Keep pull metrics independent of the business reporting backend. */
    public static FlexMonitor wrap(FlexMonitor monitor, FlexMonitor observations) {
        return new SafeFlexMonitor(
                monitor instanceof SafeFlexMonitor safe ? safe.delegate : monitor, observations);
    }

    @Override
    public <T> void observe(
            String name,
            FlexMetricType type,
            T source,
            java.util.function.ToDoubleFunction<T> value) {
        run(name, () -> observations.observe(name, type, source, value));
    }

    /** Isolate direct metric registration during component initialization. */
    public static void run(String name, Runnable metricOperation) {
        try {
            metricOperation.run();
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void register(String name, FlexMetricType type) {
        try {
            delegate.register(name, type);
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void register(String name, FlexMetricType type, FlexPriorityType priority) {
        try {
            delegate.register(name, type, priority);
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void register(String name, FlexMetricType type, int statisticsType) {
        try {
            delegate.register(name, type, statisticsType);
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void report(String name, double value) {
        try {
            delegate.report(name, value);
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void report(String name, FlexMetricTags tags, double value) {
        try {
            delegate.report(name, tags, value);
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void prepare(String name, FlexMetricTags tags) {
        try {
            delegate.prepare(name, tags);
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void removeGauge(String name, FlexMetricTags tags) {
        try {
            delegate.removeGauge(name, tags);
        } catch (Throwable failure) {
            logFailure(name, failure);
        }
    }

    @Override
    public void close() {
        try {
            delegate.close();
        } catch (Throwable failure) {
            logFailure("close", failure);
        }
    }

    static void logFailure(String name, Throwable failure) {
        try {
            log.debug("Metric operation failed: metric={}", name, failure);
        } catch (Throwable ignored) {
        }
    }
}
