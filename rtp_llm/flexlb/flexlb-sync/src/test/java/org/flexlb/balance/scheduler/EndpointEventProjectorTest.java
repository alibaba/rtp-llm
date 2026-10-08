package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.mockito.Mockito.*;

class EndpointEventProjectorTest {
    @Test
    void localPrefillTerminalCannotSettleDecodeReservation() {
        for (var kind : List.of(PrefillState.WorkerStatusFact.Kind.PRIORITY_CANCELED,
                PrefillState.WorkerStatusFact.Kind.FAILED, PrefillState.WorkerStatusFact.Kind.COMPLETED)) {
            RequestRegistry registry = mock(RequestRegistry.class);
            PrefillEndpoint prefill = mock(PrefillEndpoint.class);
            ScheduledRequest request = mock(ScheduledRequest.class);
            var fact = PrefillState.WorkerStatusFact.terminal(request, kind,
                    kind == PrefillState.WorkerStatusFact.Kind.COMPLETED ? 0L : 8429L);
            new EndpointEventProjector(registry).onPrefillStatus(prefill, RoleType.PREFILL, List.of(fact));
            verifyNoInteractions(registry);
        }
    }
    @Test
    void decodeTerminalIsTheAuthorityForRequestSettlement() {
        RequestRegistry registry = mock(RequestRegistry.class);
        RequestSlot slot = mock(RequestSlot.class);
        ScheduledRequest request = mock(ScheduledRequest.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        var reservation = new DecodeEndpoint.ReservationHandle(1L, 2L, 3L);
        when(registry.requestSlot(2L)).thenReturn(slot);
        when(registry.isCurrentSlot(slot)).thenReturn(true);
        when(slot.ownsDecodeFact(decode, reservation)).thenReturn(true);
        when(slot.activeItem()).thenReturn(request);
        new EndpointEventProjector(registry).onDecodeStatus(decode,
                List.of(DecodeEndpoint.WorkerStatusFact.terminal(reservation, 8429L)));
        verify(slot).markDecodeTerminalOwned();
        verify(slot).reduceWorkerTerminal(eq(request), any(DeferredTerminal.class));
    }
}
