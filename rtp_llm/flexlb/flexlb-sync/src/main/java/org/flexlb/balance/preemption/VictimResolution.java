package org.flexlb.balance.preemption;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Request-side resolution of one exact preemption victim: an end was selected or delivery resumed.
 * REQUEST_END does not imply that resource cleanup or request archival has completed.
 * Decode release is established separately by the endpoint ledger.
 */
public record VictimResolution(long requestId, Outcome outcome) {

    public enum Outcome { REQUEST_END, DELIVERY_RESUMED }

    public VictimResolution {
        java.util.Objects.requireNonNull(outcome, "outcome");
        checkArgument(requestId > 0, "requestId must be positive");
    }
}
