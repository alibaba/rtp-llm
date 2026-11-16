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
