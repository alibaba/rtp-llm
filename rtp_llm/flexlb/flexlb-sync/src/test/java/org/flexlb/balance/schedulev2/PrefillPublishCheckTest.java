package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.balance.schedulev2.request.Request;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillPublishCheckTest {

    private final EndpointRegistry endpoints = mock(EndpointRegistry.class);
    private final PrefillEndpoint prefill = V2TestSupport.prefillEndpoint("p1:8000");
    private final WorkerEndpoint.EndpointHold hold = mock(WorkerEndpoint.EndpointHold.class);
    private final Request request = V2TestSupport.request(SchedulingTestConfig.newConfig(), "1", 50, 16L);
    private final PrefillPublishCheck check = new PrefillPublishCheck(endpoints, mock(PrefillQueueRegistry.class));

    PrefillPublishCheckTest() {
        V2TestSupport.choose(request, prefill, 0L, 10L);
        when(endpoints.get(RoleType.PREFILL, "p1:8000")).thenReturn(prefill);
        when(endpoints.hold(RoleType.PREFILL, "p1:8000")).thenReturn(hold);
        when(hold.endpoint()).thenReturn(prefill);
    }

    @Test
    void currentPrefillReturnsHoldForPublication() {
        assertTrue(check.isAvailable(request));
        try (WorkerEndpoint.EndpointHold acquired = check.hold(request)) {
            assertSame(hold, acquired);
        }
        verify(hold).close();
    }

    @Test
    void retiringPrefillIsRejectedEvenWhileStillInRegistry() {
        when(prefill.isRetired()).thenReturn(true);
        assertFalse(check.isAvailable(request));
        assertNull(check.hold(request));
        verify(hold).close();
    }

    @Test
    void replacedPrefillAtSameAddressIsRejected() {
        PrefillEndpoint replacement = V2TestSupport.prefillEndpoint("p1:8000");
        when(endpoints.get(RoleType.PREFILL, "p1:8000")).thenReturn(replacement);
        when(hold.endpoint()).thenReturn(replacement);
        assertFalse(check.isAvailable(request));
        assertNull(check.hold(request));
        verify(hold).close();
    }

    @Test
    void removedPrefillIsRejected() {
        when(endpoints.get(RoleType.PREFILL, "p1:8000")).thenReturn(null);
        when(endpoints.hold(RoleType.PREFILL, "p1:8000")).thenReturn(null);
        assertFalse(check.isAvailable(request));
        assertNull(check.hold(request));
    }
}
