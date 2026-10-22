package org.flexlb.balance.scheduler;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.ArgumentMatchers.*;

class RequestCancellationProofTest {
    private static final DecodeEndpoint.ReservationHandle RESERVATION =
            new DecodeEndpoint.ReservationHandle(1L, 701L, 5L);

    @ParameterizedTest
    @EnumSource(EngineCancelChannel.CancelAck.class)
    void onlyOriginalPrefillFenceAcknowledgementSettlesExactDelivery(EngineCancelChannel.CancelAck ack) throws Exception {
        try (Fixture f = new Fixture()) {
            f.delivery.complete(DeliveryResult.timedOut(new RuntimeException("lost ACK")));
            verify(f.channel).cancel(new CancelTarget("127.0.0.1", 8123), 701L, 50L);
            f.cancel.complete(ack);
            f.slot.expireInactiveRequest(Long.MAX_VALUE);
            if (ack == EngineCancelChannel.CancelAck.REQUEST_FENCED) {
                assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                        f.slot.future().get(1, TimeUnit.SECONDS).getCode());
                assertEquals(RequestState.Phase.TIMED_OUT, f.slot.snapshot().state());
                verify(f.decode).release(RESERVATION, DecodeEndpoint.ReleaseReason.REQUEST_FENCED);
                verify(f.prefill).releaseCommittedItem(f.item);
            } else {
                assertFalse(f.slot.future().isDone());
                assertSame(f.item, f.slot.activeItem());
                assertTrue(f.slot.inactivityDeadlineAtMs().isEmpty(), "no expired timer spin or fabricated release proof");
                verify(f.decode, never()).release(any(), any());
                verify(f.prefill, never()).releaseCommittedItem(any());
            }
        }
    }

    @Test
    void transportFailureRetainsTheExactRequestUntilWorkerTerminal() throws Exception {
        try (Fixture f = new Fixture()) {
            f.delivery.complete(DeliveryResult.uncertain(new RuntimeException("connection reset after send")));
            f.cancel.completeExceptionally(new IllegalStateException("Cancel transport unknown"));
            f.slot.expireInactiveRequest(Long.MAX_VALUE);
            assertFalse(f.slot.future().isDone());
            assertSame(f.item, f.slot.activeItem());
            f.slot.processDecodeStatus(f.decode, DecodeEndpoint.WorkerStatusFact.terminal(RESERVATION, 42L));
            assertFalse(f.slot.future().get(1, TimeUnit.SECONDS).isSuccess());
            assertEquals(RequestState.Phase.FAILED, f.slot.snapshot().state());
            verify(f.decode, never()).release(any(), eq(DecodeEndpoint.ReleaseReason.REQUEST_FENCED));
        }
    }

    @Test
    void lateFenceCannotChangeAnAlreadySettledExactGeneration() throws Exception {
        try (Fixture f = new Fixture()) {
            f.delivery.complete(DeliveryResult.timedOut(new RuntimeException("lost ACK")));
            f.slot.processDecodeStatus(f.decode, DecodeEndpoint.WorkerStatusFact.terminal(RESERVATION, 42L));
            Response before = f.slot.future().get(1, TimeUnit.SECONDS);
            RequestState state = f.slot.snapshot();
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
            assertSame(before, f.slot.future().join());
            assertEquals(state, f.slot.snapshot());
            verify(f.decode, never()).release(any(), eq(DecodeEndpoint.ReleaseReason.REQUEST_FENCED));
        }
    }

    private static final class Fixture implements AutoCloseable {
        final EngineCancelChannel channel = mock(EngineCancelChannel.class);
        final CompletableFuture<EngineCancelChannel.CancelAck> cancel = new CompletableFuture<>();
        final DecodeEndpoint decode = mock(DecodeEndpoint.class);
        final PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        final RequestCompletionPublisher publisher = new RequestCompletionPublisher(1, null);
        final RequestSlot slot;
        final ScheduledRequest item;
        final RequestSlot.DeliveryClaim delivery;
        Fixture() {
            var config = new FlexlbConfig();
            var context = new BalanceContext(config);
            var request = new Request(); request.setRequestId(701L); context.setRequest(request);
            var timer = mock(ExpirationTimer.class);
            slot = new RequestSlot(publisher, context, timer, new RequestTerminalCleanup(timer),
                    () -> { }, true, null, channel);
            var server = new ServerStatus(); server.setServerIp("127.0.0.1"); server.setGrpcPort(8123);
            item = new ScheduledRequest(context, slot.future(), new Response(), server, null,
                    prefill, decode, RESERVATION, System.currentTimeMillis());
            slot.configureInactivityTimeout(1800L);
            var admission = slot.tryBeginAdmissionHandle(); assertNotNull(admission);
            assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS, slot.commitRoute(item, () -> true));
            admission.close();
            when(channel.cancel(any(), anyLong(), anyLong())).thenReturn(cancel);
            delivery = slot.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, 11L, () -> true);
            assertNotNull(delivery);
        }
        @Override public void close() { publisher.close(); }
    }
}
