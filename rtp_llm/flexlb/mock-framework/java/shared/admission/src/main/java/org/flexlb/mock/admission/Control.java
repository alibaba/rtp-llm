package org.flexlb.mock.admission;

import org.flexlb.mock.engine.Scheduler;

/** Generation protocol permits; these are not universal Scheduler states. */
public enum Control implements Scheduler.Control {
    GENERATE, TRANSFER_ACKNOWLEDGED, PREFILL_COMPLETE
}
