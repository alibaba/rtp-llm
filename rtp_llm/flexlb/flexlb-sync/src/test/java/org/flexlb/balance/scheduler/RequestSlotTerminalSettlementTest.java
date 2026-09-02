package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class RequestSlotTerminalSettlementTest {

    private static final DecodeEndpoint.ReservationHandle RESERVATION =
            new DecodeEndpoint.ReservationHandle(1L, 2L, 3L);

    @Test
    void decodeTerminalIsAProofOfAlreadyCommittedEndpointSettlement() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DeferredTerminal terminal = DeferredTerminal.worker(
                new WorkerTerminalObservation(
                        WorkerTerminalSource.DECODE_ENDPOINT_SETTLED,
                        true,
                        0L));

        assertTrue(RequestSlot.terminalOwnsDecodeSettlement(
                terminal, decode, 4L, RESERVATION));
        verifyNoInteractions(decode);
    }

    @Test
    void prefillBackedTerminalDelegatesToTheExactDecodeClaimTransaction() {
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        DeferredTerminal terminal = DeferredTerminal.worker(
                new WorkerTerminalObservation(
                        WorkerTerminalSource.PREFILL_BACKED,
                        false,
                        9L));
        when(decode.reconcilePriorityVictimFinished(4L, RESERVATION))
                .thenReturn(false);

        assertFalse(RequestSlot.terminalOwnsDecodeSettlement(
                terminal, decode, 4L, RESERVATION));
        verify(decode).reconcilePriorityVictimFinished(4L, RESERVATION);
    }
}
