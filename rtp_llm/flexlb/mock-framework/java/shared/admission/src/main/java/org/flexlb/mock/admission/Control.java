package org.flexlb.mock.admission;

import org.flexlb.mock.engine.Scheduler;

/** Generation protocol permits; these are not universal Scheduler states. */
public enum Control implements Scheduler.Control {
    TRANSFER_ACKNOWLEDGED, PREFILL_COMPLETE;

    /** The native first sampled token is also the repeated synthetic Decode output. */
    public record Generate(int tokenId) implements Scheduler.Control {}
}
