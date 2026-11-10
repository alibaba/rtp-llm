package org.flexlb.balance.preemption;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Reports whether a request selected for preemption will end or resume delivery.
 * REQUEST_END does not imply that resource cleanup or request archival has completed.
 * Decode release is established separately by the endpoint ledger.
 */
public record PreemptionRequestResult(String requestId, Outcome outcome) {
    public PreemptionRequestResult {
        java.util.Objects.requireNonNull(outcome, "outcome");
        checkArgument(requestId != null && !requestId.isBlank(), "requestId is required");
    }

    public enum Outcome {
        REQUEST_END,
        DELIVERY_RESUMED
    }
}
