package org.flexlb.balance.scheduler;

import java.util.Objects;

/** Immutable terminal intention, selected before cleanup and committed only after cleanup. */
record TerminalOutcome(RequestState.Phase phase, String detail) {
    TerminalOutcome {
        Objects.requireNonNull(phase, "phase");
        if (!phase.isTerminal()) { throw new IllegalArgumentException("terminal phase required"); }
    }
    static TerminalOutcome fail(String detail) { return new TerminalOutcome(RequestState.Phase.FAILED, detail); }
    static TerminalOutcome complete(String detail) { return new TerminalOutcome(RequestState.Phase.COMPLETED, detail); }
    static TerminalOutcome cancel(String detail) { return new TerminalOutcome(RequestState.Phase.CANCELLED, detail); }
    static TerminalOutcome timeout(String detail) { return new TerminalOutcome(RequestState.Phase.TIMED_OUT, detail); }
    static TerminalOutcome cancellation(CancelReason reason, String detail) {
        return reason == CancelReason.DEADLINE_EXCEEDED ? timeout(detail) : cancel(detail);
    }
}
