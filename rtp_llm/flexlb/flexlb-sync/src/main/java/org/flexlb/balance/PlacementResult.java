package org.flexlb.balance;

import org.flexlb.dao.loadbalance.Response;

import java.util.Map;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * One placement attempt and its internal diagnostic snapshot, separate from the wire response.
 */
public record PlacementResult<V, B>(
        Status status, V value, Response failure, B blocker, Map<String, Object> diagnostics) {
    public PlacementResult {
        Objects.requireNonNull(status, "status");
        diagnostics = diagnostics == null ? null : Map.copyOf(diagnostics);
        boolean success = status == Status.SUCCESS;
        boolean rejected = status == Status.REJECTED;
        boolean blocked = status == Status.BLOCKED;
        checkArgument(
                success == (value != null)
                        && (!rejected || failure != null)
                        && (rejected || blocked || status == Status.RETRY || failure == null)
                        && blocked == (blocker != null),
                "placement status requires its exact payload");
        checkArgument(
                failure == null || !failure.isSuccess(),
                "placement failure requires a failure response");
    }

    public static <V, B> PlacementResult<V, B> success(V value) {
        return new PlacementResult<>(
                Status.SUCCESS, Objects.requireNonNull(value, "value"), null, null, null);
    }

    public static <V, B> PlacementResult<V, B> rejected(Response response) {
        return rejected(response, null);
    }

    public static <V, B> PlacementResult<V, B> rejected(
            Response response, Map<String, Object> diagnostics) {
        return new PlacementResult<>(
                Status.REJECTED,
                null,
                Objects.requireNonNull(response, "response"),
                null,
                diagnostics);
    }

    public static <V, B> PlacementResult<V, B> blocked(B blocker) {
        return new PlacementResult<>(
                Status.BLOCKED, null, null, Objects.requireNonNull(blocker, "blocker"), null);
    }

    /**
     * A retryable block retains the failure observed by the same decision.
     */
    public static <V, B> PlacementResult<V, B> blocked(
            B blocker, Response failure, Map<String, Object> diagnostics) {
        return new PlacementResult<>(
                Status.BLOCKED,
                null,
                failure,
                Objects.requireNonNull(blocker, "blocker"),
                diagnostics);
    }

    public static <V, B> PlacementResult<V, B> closed() {
        return new PlacementResult<>(Status.CLOSED, null, null, null, null);
    }

    @SuppressWarnings("unchecked")
    public <T> PlacementResult<T, B> asFailure() {
        checkArgument(status != Status.SUCCESS, "successful placement has a value");
        return (PlacementResult<T, B>) this;
    }

    public static <V, B> PlacementResult<V, B> retry() {
        return new PlacementResult<>(Status.RETRY, null, null, null, null);
    }

    public static <V, B> PlacementResult<V, B> retry(Response failure) {
        return new PlacementResult<>(Status.RETRY, null, failure, null, null);
    }

    /**
     * One atomic attempt to register the selected endpoint's request impact.
     */
    public enum Registration {
        REGISTERED,
        CONFLICT,
        BLOCKED,
        UNAVAILABLE
    }

    public enum Status {
        RETRY,
        SUCCESS,
        REJECTED,
        BLOCKED,
        CLOSED
    }
}
