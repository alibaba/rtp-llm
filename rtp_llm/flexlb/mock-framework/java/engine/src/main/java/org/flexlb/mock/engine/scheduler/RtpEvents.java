package org.flexlb.mock.engine.scheduler;

/** RTP events accepted by the scheduler across the execution boundary. */
public final class RtpEvents {
    private RtpEvents() {}

    public enum Control implements Scheduler.Control {
        TRANSFER_ACKNOWLEDGED, PREFILL_COMPLETE;

        /** The first sampled token is also the repeated synthetic Decode output. */
        public record Generate(int tokenId) implements Scheduler.Control {}
    }

    public record TransferResult(boolean prefillFinished, RequestFailure failure) implements Scheduler.TransferResult {}
}
