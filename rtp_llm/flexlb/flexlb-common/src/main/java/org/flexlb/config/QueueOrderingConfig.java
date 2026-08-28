package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

import java.util.Optional;

/** Queue ordering mode and optional priority/preemption settings. */
@Getter
@Setter
public final class QueueOrderingConfig {

    public enum Type {
        FIFO,
        PRIORITY
    }

    private Type type = Type.FIFO;
    private int defaultPriority = 50;
    private PreemptionConfig preemption;

    public static QueueOrderingConfig priority() {
        QueueOrderingConfig config = new QueueOrderingConfig();
        config.type = Type.PRIORITY;
        return config;
    }

    public Optional<PreemptionConfig> preemptionPolicy() {
        return type == Type.PRIORITY
                ? Optional.ofNullable(preemption)
                : Optional.empty();
    }
}
