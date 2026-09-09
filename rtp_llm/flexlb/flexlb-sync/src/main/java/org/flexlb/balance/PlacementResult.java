package org.flexlb.balance;

import org.flexlb.dao.loadbalance.Response;

import java.util.Objects;

/** One placement attempt: success, capacity block, rejection, or closed owner. */
public record PlacementResult<V, B>(
        Status status,
        V value,
        Response rejection,
        B blocker) {

    public PlacementResult {
        Objects.requireNonNull(status, "status");
        boolean success = status == Status.SUCCESS;
        boolean rejected = status == Status.REJECTED;
        boolean blocked = status == Status.BLOCKED;
        if (success != (value != null)
                || rejected != (rejection != null)
                || blocked != (blocker != null)) {
            throw new IllegalArgumentException(
                    "placement status requires its exact payload");
        }
        if (rejection != null && rejection.isSuccess()) {
            throw new IllegalArgumentException(
                    "placement rejection requires a failure response");
        }
    }

    public static <V, B> PlacementResult<V, B> success(V value) {
        return new PlacementResult<>(Status.SUCCESS,
                Objects.requireNonNull(value, "value"), null, null);
    }

    public static <V, B> PlacementResult<V, B> rejected(Response response) {
        return new PlacementResult<>(Status.REJECTED, null,
                Objects.requireNonNull(response, "response"), null);
    }

    public static <V, B> PlacementResult<V, B> blocked(B blocker) {
        return new PlacementResult<>(Status.BLOCKED, null, null,
                Objects.requireNonNull(blocker, "blocker"));
    }

    public static <V, B> PlacementResult<V, B> closed() {
        return status(Status.CLOSED);
    }

    public static <V, B> PlacementResult<V, B> limitReached() {
        return status(Status.LIMIT_REACHED);
    }

    private static <V, B> PlacementResult<V, B> status(Status status) {
        return new PlacementResult<>(status, null, null, null);
    }

    public enum Status {
        SUCCESS,
        REJECTED,
        BLOCKED,
        CLOSED,
        LIMIT_REACHED
    }
}
