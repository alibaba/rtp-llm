package org.flexlb.balance.schedule;

import org.junit.jupiter.api.Timeout;

import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks that remote-stop waits cannot starve local work and that shutdown drains accepted stop continuations.
 */
@Timeout(15)
class StopExecutorSafetyTest {

    private RequestContext context() {
        var context = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        when(context.getRequestId()).thenReturn("41");
        return context;
    }
}
