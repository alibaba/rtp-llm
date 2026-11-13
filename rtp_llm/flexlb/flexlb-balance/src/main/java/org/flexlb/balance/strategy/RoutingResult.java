package org.flexlb.balance.strategy;

import org.flexlb.worker.endpoint.Response;

import java.util.Map;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * One routing result and its internal diagnostic snapshot, separate from the wire response.
 */
public record RoutingResult<V, B>(
        Status status, V value, Response failure, B blocker, Map<String, Object> diagnostics) {
    public RoutingResult {
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
                "routing status requires its exact payload");
        checkArgument(
                failure == null || !failure.isSuccess(),
                "routing failure requires a failure response");
    }

    public static <V, B> RoutingResult<V, B> success(V value) {
        return new RoutingResult<>(Status.SUCCESS, value, null, null, null);
    }

    public static <V, B> RoutingResult<V, B> rejected(Response response) {
        return rejected(response, null);
    }

    public static <V, B> RoutingResult<V, B> rejected(
            Response response, Map<String, Object> diagnostics) {
        return new RoutingResult<>(Status.REJECTED, null, response, null, diagnostics);
    }

    public static <V, B> RoutingResult<V, B> blocked(B blocker) {
        return new RoutingResult<>(Status.BLOCKED, null, null, blocker, null);
    }

    /**
     * A retryable block retains the failure observed by the same decision.
     */
    public static <V, B> RoutingResult<V, B> blocked(
            B blocker, Response failure, Map<String, Object> diagnostics) {
        return new RoutingResult<>(Status.BLOCKED, null, failure, blocker, diagnostics);
    }

    public static <V, B> RoutingResult<V, B> closed() {
        return new RoutingResult<>(Status.CLOSED, null, null, null, null);
    }

    @SuppressWarnings("unchecked")
    public <T> RoutingResult<T, B> asFailure() {
        checkArgument(status != Status.SUCCESS, "successful placement has a value");
        return (RoutingResult<T, B>) this;
    }

    public static <V, B> RoutingResult<V, B> retry() {
        return new RoutingResult<>(Status.RETRY, null, null, null, null);
    }

    public static <V, B> RoutingResult<V, B> retry(Response failure) {
        return new RoutingResult<>(Status.RETRY, null, failure, null, null);
    }

    public enum Status {
        RETRY,
        SUCCESS,
        REJECTED,
        BLOCKED,
        CLOSED
    }
}
