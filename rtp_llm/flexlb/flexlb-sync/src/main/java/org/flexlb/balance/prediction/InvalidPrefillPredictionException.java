package org.flexlb.balance.prediction;

/** A predictor returned a numeric value outside the scheduling contract. */
public final class InvalidPrefillPredictionException extends IllegalStateException {

    InvalidPrefillPredictionException(
            String predictionKind, Number returnedValue) {
        super("Prefill predictor returned invalid " + predictionKind
                + " milliseconds: " + returnedValue);
    }
}
