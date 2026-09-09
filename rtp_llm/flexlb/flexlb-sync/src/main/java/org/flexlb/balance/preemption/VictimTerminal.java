package org.flexlb.balance.preemption;

/** Authoritative terminal proof for one exact preemption victim. */
public record VictimTerminal(long requestId) {

    public VictimTerminal {
        if (requestId <= 0) {
            throw new IllegalArgumentException(
                    "requestId must be positive");
        }
    }
}
