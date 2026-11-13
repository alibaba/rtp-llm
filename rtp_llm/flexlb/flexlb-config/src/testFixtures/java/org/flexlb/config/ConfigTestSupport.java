package org.flexlb.config;

import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.QueueOrderingConfig;
import org.flexlb.config.scheduler.SchedulerConfig;

/**
 * Configuration factories used only by test fixtures.
 */
public final class ConfigTestSupport {
    private ConfigTestSupport() {}

    public static SchedulerConfig direct() {
        SchedulerConfig config = new SchedulerConfig();
        config.setType(SchedulerConfig.Type.DIRECT);
        return config;
    }

    public static QueueOrderingConfig priority() {
        QueueOrderingConfig config = new QueueOrderingConfig();
        config.setType(QueueOrderingConfig.Type.PRIORITY);
        return config;
    }

    public static java.util.Optional<PreemptionConfig> preemptionPolicy(
            QueueOrderingConfig ordering) {
        return ordering.getType() == QueueOrderingConfig.Type.PRIORITY
                ? java.util.Optional.ofNullable(ordering.getPreemption())
                : java.util.Optional.empty();
    }
}
