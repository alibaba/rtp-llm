package org.flexlb.balance.preemption;

import java.util.concurrent.CompletionStage;

/** Exact token-fenced ownership of one Engine preemption victim. */
public interface PreemptionClaim {

    long requestId();

    long attemptToken();

    /** Read-only observation of the canonical victim terminal. */
    CompletionStage<VictimTerminal> terminalObservation();
}
