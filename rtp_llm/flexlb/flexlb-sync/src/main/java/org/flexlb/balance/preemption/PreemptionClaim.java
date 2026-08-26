package org.flexlb.balance.preemption;

import java.util.concurrent.CompletableFuture;

/** Exact token-fenced ownership of one Engine preemption victim. */
public interface PreemptionClaim {

    long requestId();

    long attemptToken();

    CompletableFuture<VictimTerminal> terminal();
}
