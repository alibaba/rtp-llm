package org.flexlb.balance.prediction;

/** A predictor returned a numeric value outside the scheduling contract. */
public final class InvalidPrefillPredictionException extends IllegalStateException {

    public enum PredictionKind {
        SINGLE_REQUEST,
        DECISION_GROUP
    }

    private final PredictionKind predictionKind;
    private final Number returnedValue;

    InvalidPrefillPredictionException(
            PredictionKind predictionKind, Number returnedValue) {
        super("Prefill predictor returned invalid " + predictionKind
                + " milliseconds: " + returnedValue);
        this.predictionKind = predictionKind;
        this.returnedValue = returnedValue;
    }

    public PredictionKind predictionKind() {
        return predictionKind;
    }

    public Number returnedValue() {
        return returnedValue;
    }
}
