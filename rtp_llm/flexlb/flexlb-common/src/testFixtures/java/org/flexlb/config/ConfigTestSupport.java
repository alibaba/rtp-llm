package org.flexlb.config;

/** Configuration factories used only by test fixtures. */
public final class ConfigTestSupport {
    private ConfigTestSupport() {}

    public static DispatcherConfig nonBatch() {
        DispatcherConfig config = new DispatcherConfig();
        config.setType(DispatcherConfig.Type.NON_BATCH);
        return config;
    }

    public static DecisionPolicyConfig single() {
        DecisionPolicyConfig config = new DecisionPolicyConfig();
        config.setType(DecisionPolicyConfig.Type.SINGLE);
        return config;
    }

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

    public static java.util.Optional<PreemptionConfig> preemptionPolicy(QueueOrderingConfig ordering) {
        return ordering.getType() == QueueOrderingConfig.Type.PRIORITY
            ? java.util.Optional.ofNullable(ordering.getPreemption()) : java.util.Optional.empty();
    }

    public static boolean isSingleDecision(FlexlbConfig config) {
        return config.isQueue() && config.decisionPolicy().getType() == DecisionPolicyConfig.Type.SINGLE;
    }
}
