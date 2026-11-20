package org.flexlb.balance.strategy;

import org.flexlb.balance.request.Response;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks routing result payloads through both factories and direct construction.
 */
class SelectionResultTest {
    @Test
    void retryHintsBelongToTheBlockedResultAndSurviveRouteFailurePropagation() {
        var candidates = new java.util.HashSet<Long>();
        candidates.add(7L);
        var blocked = SelectionResult.blocked(new Object()).withCandidates(candidates);
        candidates.clear();
        org.junit.jupiter.api.Assertions.assertEquals(
                java.util.Set.of(7L), blocked.candidateGenerations());
        assertSame(blocked, blocked.asFailure());
        assertThrows(
                UnsupportedOperationException.class, () -> blocked.candidateGenerations().clear());
        assertThrows(
                IllegalArgumentException.class,
                () -> SelectionResult.success(new Object()).withCandidates(java.util.Set.of(7L)));
    }

    @Test
    void factoriesCannotCreateResultsMissingTheirRequiredPayload() {
        assertThrows(IllegalArgumentException.class, () -> SelectionResult.success(null));
        assertThrows(IllegalArgumentException.class, () -> SelectionResult.rejected(null));
        assertThrows(IllegalArgumentException.class, () -> SelectionResult.blocked(null));
        assertThrows(
                IllegalArgumentException.class, () -> SelectionResult.blocked(null, null, null));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new SelectionResult<>(
                                SelectionResult.Status.SUCCESS, null, null, null, null));
    }

    @Test
    void factoriesKeepTheExactAcceptedPayload() {
        var selected = new Object();
        var blocker = new Object();
        var failure = Response.error(org.flexlb.worker.endpoint.StrategyErrorType.INVALID_REQUEST);
        assertSame(selected, SelectionResult.success(selected).value());
        assertSame(blocker, SelectionResult.blocked(blocker).blocker());
        assertSame(failure, SelectionResult.rejected(failure).failure());
    }
}
