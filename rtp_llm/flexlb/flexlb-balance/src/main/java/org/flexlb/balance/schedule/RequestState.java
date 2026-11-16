package org.flexlb.balance.schedule;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Objects;

/**
 * Immutable request state used by live queries and retained after cleanup.
 */
public record RequestState(
        @JsonProperty("requestId") String requestId,
        @JsonProperty("originalNumericRequestId") long originalNumericRequestId,
        @JsonProperty("requestPhase") RequestPhase requestPhase,
        @JsonProperty("state") State state,
        @JsonProperty("createdAtMs") long createdAtMs,
        @JsonProperty("updatedAtMs") long updatedAtMs,
        @JsonProperty("detail") String detail) {
    public RequestState {
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(detail, "detail");
    }

    public static RequestState from(RequestContext.Snapshot snapshot) {
        if (snapshot.requestRecordForArchive() != null) {
            return snapshot.requestRecordForArchive();
        }
        if (snapshot.requestId() == null) {
            return null;
        }
        var delivery = snapshot.committedRoute();
        var cancellation = snapshot.cancellationReason();
        State phase =
                snapshot.endState() != null
                        ? snapshot.endState()
                        : cancellation != null
                                ? State.CANCEL_REQUESTED
                                : delivery != null ? State.ACKNOWLEDGED : State.QUEUED;
        String detail =
                snapshot.endState() != null
                        ? snapshot.endDetail()
                        : cancellation != null
                                ? snapshot.cancellationDetail()
                                : snapshot.visibilityOverdue()
                                        ? "SUSPECTED_LOST: request not visible before its deadline"
                                        : delivery != null && snapshot.responseForFuture() != null
                                                ? "delivery acknowledged"
                                                : "queued";
        return new RequestState(
                snapshot.requestId(),
                snapshot.originalNumericRequestId(),
                snapshot.requestPhase(),
                phase,
                snapshot.createdAtMs(),
                snapshot.updatedAtMs(),
                detail == null ? "" : detail);
    }

    /**
     * Reports request progress or its final outcome; a terminal outcome does not imply cleanup is finished.
     */
    public enum State {
        QUEUED,
        ACKNOWLEDGED,
        CANCEL_REQUESTED,
        CANCELLED,
        TIMED_OUT,
        FAILED,
        COMPLETED;

        public boolean isTerminal() {
            return this == CANCELLED || this == TIMED_OUT || this == FAILED || this == COMPLETED;
        }
    }
}
