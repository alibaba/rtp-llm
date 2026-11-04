package org.flexlb.mock.admission;

import org.flexlb.mock.engine.RequestFailure;
import org.flexlb.mock.engine.Scheduler;

/**
 * Events passed from the RTP protocol handlers to the scheduler.
 */
public final class RtpEvents {
    private RtpEvents() {}

    public enum Control implements Scheduler.Control {
        TRANSFER_ACKNOWLEDGED, PREFILL_COMPLETE;

        public record Generate(int tokenId) implements Scheduler.Control {}
    }

    public record TransferResult(boolean prefillFinished, RequestFailure failure) implements Scheduler.TransferResult {}
}
