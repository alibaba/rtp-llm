package org.flexlb.balance.delivery;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Checks that capacity results retain their exact value or rejection without incomplete payloads.
 */
class CapacityBoundaryTest {
    @Test
    void factoriesCannotCreateIncompleteCapacityResults() {
        assertThrows(IllegalArgumentException.class, () -> CapacityBoundary.unavailable(null));
        assertThrows(IllegalArgumentException.class, () -> CapacityBoundary.failed(null));
        assertThrows(IllegalArgumentException.class, () -> CapacityBoundary.Attempt.accepted(null));
        assertThrows(IllegalArgumentException.class, () -> CapacityBoundary.Attempt.rejected(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> new CapacityBoundary(CapacityBoundary.Status.FAILED, null, null));
    }

    @Test
    void acceptedAndRejectedAttemptsRetainTheirExactValue() {
        var value = new Object();
        var cause = new IllegalStateException("capacity check failed");
        var boundary = CapacityBoundary.failed(cause);
        assertSame(value, CapacityBoundary.Attempt.accepted(value).value());
        assertSame(cause, boundary.cause());
        assertSame(boundary, CapacityBoundary.Attempt.rejected(boundary).boundary());
    }
}
