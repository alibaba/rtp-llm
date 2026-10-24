package org.flexlb.balance.scheduler.decode;

import org.flexlb.config.FlexlbConfig;

/**
 * What a request does when no Decode worker can accept it.
 */
public enum DecodeMode {
    IMMEDIATE, WAIT_AT_PLACEMENT, PREEMPT_AT_PLACEMENT;

    public static DecodeMode from(FlexlbConfig config) {
        return config.queueScheduler().getOrdering().preemptionPolicy().isPresent()
                ? PREEMPT_AT_PLACEMENT
                : WAIT_AT_PLACEMENT;
    }
}
