package org.flexlb.balance.strategy;

import org.flexlb.worker.endpoint.Response;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks routing result payloads through both factories and direct construction.
 */
class RoutingResultTest {
    @Test
    void factoriesCannotCreateResultsMissingTheirRequiredPayload() {
        assertThrows(IllegalArgumentException.class, () -> RoutingResult.success(null));
        assertThrows(IllegalArgumentException.class, () -> RoutingResult.rejected(null));
        assertThrows(IllegalArgumentException.class, () -> RoutingResult.blocked(null));
        assertThrows(IllegalArgumentException.class, () -> RoutingResult.blocked(null, null, null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new RoutingResult<>(RoutingResult.Status.SUCCESS, null, null, null, null));
    }

    @Test
    void factoriesKeepTheExactAcceptedPayload() {
        var selected = new Object();
        var blocker = new Object();
        var failure = Response.error(org.flexlb.worker.endpoint.StrategyErrorType.INVALID_REQUEST);
        assertSame(selected, RoutingResult.success(selected).value());
        assertSame(blocker, RoutingResult.blocked(blocker).blocker());
        assertSame(failure, RoutingResult.rejected(failure).failure());
    }
}
