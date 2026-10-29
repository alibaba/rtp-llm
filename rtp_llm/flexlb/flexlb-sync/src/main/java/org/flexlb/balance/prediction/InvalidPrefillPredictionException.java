package org.flexlb.balance.prediction;

/**
 * A predictor returned a numeric value outside the scheduling contract.
 */
public final class InvalidPrefillPredictionException extends IllegalStateException {
    private final int memberIndex;

    InvalidPrefillPredictionException(String predictionKind, Number returnedValue) {
        super(
                "Prefill predictor returned invalid "
                        + predictionKind
                        + " milliseconds: "
                        + returnedValue);
        memberIndex = 0;
    }

    private InvalidPrefillPredictionException(InvalidPrefillPredictionException cause, int index) {
        super(cause.getMessage(), cause);
        memberIndex = index;
    }

    /**
     * Position in the ordered decision group; no request or input is retained by the failure.
     */
    public InvalidPrefillPredictionException atMember(int index) {
        return new InvalidPrefillPredictionException(this, index);
    }

    public int memberIndex() {
        return memberIndex;
    }
}
