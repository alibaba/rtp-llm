package org.flexlb.balance.preemption;

/** Authoritative terminal proof for one exact preemption victim. */
public record VictimTerminal(String requestId) {

    public VictimTerminal {
        if (requestId == null || requestId.isBlank()) {
            throw new IllegalArgumentException(
                    "requestId must not be blank");
        }
    }

}
