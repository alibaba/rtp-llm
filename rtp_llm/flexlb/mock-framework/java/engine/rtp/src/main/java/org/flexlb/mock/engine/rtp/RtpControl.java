package org.flexlb.mock.engine.rtp;

import org.flexlb.mock.engine.Scheduler;

/** Native RTP permits; these are not universal Scheduler states. */
enum RtpControl implements Scheduler.Control {
    GENERATE, OUTPUT_WRITABLE, TRANSFER_ACKNOWLEDGED, PREFILL_COMPLETE
}
