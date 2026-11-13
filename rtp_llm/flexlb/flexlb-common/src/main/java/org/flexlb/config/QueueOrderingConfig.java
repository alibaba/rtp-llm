package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Queue ordering mode and optional priority/preemption settings. */
@Getter
@Setter
public final class QueueOrderingConfig {

    public enum Type {
        FIFO,
        PRIORITY
    }

    private Type type = Type.FIFO;
    /** Prefer higher input-token benefit per new compute token within business priority. */
    private boolean prefillCostOrder;
    private int defaultPriority = 50;
    private PreemptionConfig preemption;

    public void setType(Type type) {
        this.type = type;
        if (type == Type.PRIORITY && preemption == null) {
            preemption = new PreemptionConfig();
        }
    }

}
