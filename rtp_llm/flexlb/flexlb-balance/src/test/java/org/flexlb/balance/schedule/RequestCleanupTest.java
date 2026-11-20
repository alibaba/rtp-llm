package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Verifies that cleanup notifications arriving during execution are retained and coalesced.
 */
class RequestCleanupTest {
    @Test
    void notificationBeforeExecutorExitRequiresAnotherPass() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 91001L);
        assertTrue(context.requestCleanup(null));
        assertFalse(context.requestCleanup(null));
        assertFalse(context.requestCleanup(null));
        assertFalse(context.finishCleanupPass());
        assertTrue(context.finishCleanupPass());
        assertTrue(context.requestCleanup(null));
        assertTrue(context.finishCleanupPass());
    }

    @Test
    void responseArrivingDuringCleanupIsRetainedForPublication() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 91002L);
        var response = mock(ResponseCompletionExecutor.Registration.class);
        assertTrue(context.requestCleanup(null));
        assertNull(context.takeCleanupResponse());
        assertFalse(context.requestCleanup(response));
        assertFalse(context.requestCleanup(null));
        assertFalse(context.finishCleanupPass());
        assertSame(response, context.takeCleanupResponse());
        assertNull(context.takeCleanupResponse());
        assertTrue(context.finishCleanupPass());
    }

    @Test
    void rejectedSubmissionReturnsResponseAndAllowsCleanupRetry() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), 91003L);
        var response = mock(ResponseCompletionExecutor.Registration.class);
        assertTrue(context.requestCleanup(null));
        assertFalse(context.requestCleanup(response));
        assertSame(response, context.rejectCleanup());
        assertNull(context.takeCleanupResponse());
        assertTrue(context.requestCleanup(null));
        assertTrue(context.finishCleanupPass());
    }
}
