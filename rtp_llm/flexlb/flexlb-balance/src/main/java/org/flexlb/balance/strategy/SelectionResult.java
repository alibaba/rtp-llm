package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Response;

import java.util.Map;
import java.util.Objects;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Result of one selector call: a selected value, a capacity wait, a retry, or a rejection.
 * Successful values describe the selected role or request route; this result does not track request completion.
 */
public record SelectionResult<V, B>(
        Status status,
        V value,
        Response failure,
        B blocker,
        Map<String, Object> diagnostics,
        Set<Long> candidateGenerations) {
    public SelectionResult(
            Status status, V value, Response failure, B blocker, Map<String, Object> diagnostics) {
        this(status, value, failure, blocker, diagnostics, Set.of());
    }

    public SelectionResult {
        Objects.requireNonNull(status, "status");
        diagnostics = diagnostics == null ? null : Map.copyOf(diagnostics);
        candidateGenerations = Set.copyOf(candidateGenerations);
        boolean success = status == Status.SUCCESS;
        boolean rejected = status == Status.REJECTED;
        boolean blocked = status == Status.BLOCKED;
        checkArgument(
                success == (value != null)
                        && (!rejected || failure != null)
                        && (rejected || blocked || status == Status.RETRY || failure == null)
                        && blocked == (blocker != null),
                "selection status requires its exact payload");
        checkArgument(
                failure == null || !failure.isSuccess(),
                "selection failure requires a failure response");
        checkArgument(
                blocked || candidateGenerations.isEmpty(),
                "candidate hints require a blocked selection");
    }

    /**
     * Retains the candidates considered by this blocked selection; later calls select again.
     */
    public SelectionResult<V, B> withCandidates(Set<Long> generations) {
        checkArgument(status == Status.BLOCKED, "candidate hints require a blocked selection");
        return new SelectionResult<>(status, value, failure, blocker, diagnostics, generations);
    }

    public static <V, B> SelectionResult<V, B> success(V value) {
        return new SelectionResult<>(Status.SUCCESS, value, null, null, null);
    }

    public static <V, B> SelectionResult<V, B> rejected(Response response) {
        return rejected(response, null);
    }

    public static <V, B> SelectionResult<V, B> rejected(
            Response response, Map<String, Object> diagnostics) {
        return new SelectionResult<>(Status.REJECTED, null, response, null, diagnostics);
    }

    public static <V, B> SelectionResult<V, B> blocked(B blocker) {
        return new SelectionResult<>(Status.BLOCKED, null, null, blocker, null);
    }

    /**
     * A retryable block retains the failure observed by the same decision.
     */
    public static <V, B> SelectionResult<V, B> blocked(
            B blocker, Response failure, Map<String, Object> diagnostics) {
        return new SelectionResult<>(Status.BLOCKED, null, failure, blocker, diagnostics);
    }

    public static <V, B> SelectionResult<V, B> closed() {
        return new SelectionResult<>(Status.CLOSED, null, null, null, null);
    }

    @SuppressWarnings("unchecked")
    public <T> SelectionResult<T, B> asFailure() {
        checkArgument(status != Status.SUCCESS, "successful selection has a value");
        return (SelectionResult<T, B>) this;
    }

    public static <V, B> SelectionResult<V, B> retry() {
        return new SelectionResult<>(Status.RETRY, null, null, null, null);
    }

    public static <V, B> SelectionResult<V, B> retry(Response failure) {
        return new SelectionResult<>(Status.RETRY, null, failure, null, null);
    }

    public enum Status {
        RETRY,
        SUCCESS,
        REJECTED,
        BLOCKED,
        CLOSED
    }
}
