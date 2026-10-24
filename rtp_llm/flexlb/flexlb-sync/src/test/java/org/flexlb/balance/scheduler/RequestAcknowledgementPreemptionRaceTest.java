package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class RequestAcknowledgementPreemptionRaceTest {
    @Test
    void laterPreemptionCannotDiscardAlreadyConfirmedAcknowledgementTask() throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        var context = RequestProtocolTestSupport.context(config, 9741L);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var owner = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
        SchedulerTestSupport.responseCompletions(owner.runtime).close();
        var publisher = new ResponseCompletionExecutor(1);
        ReflectionTestUtils.setField(owner.runtime, "responseCompletions", publisher);
        ReflectionTestUtils.setField(owner.runtime, "timer", mock(ExpirationTimer.class));
        RequestProtocolTestSupport.register(owner, context);
        var reservation = new DecodeResources.ReservationHandle(1L, 9741L, 3L);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var prefill = mock(PrefillEndpoint.class);
        var server = new ServerStatus();
        server.setServerIp("127.0.0.1");
        server.setGrpcPort(8090);
        var route = SchedulingTestConfig.createRoute(context, new Response(), server, null,
                prefill, decode, reservation, System.currentTimeMillis());
        var admission = RequestProtocolTestSupport.beginAdmission(owner, context);
        assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                owner.commitRoute(route, RequestProtocolTestSupport.publication(() -> true)));
        admission.finish();
        var delivery = owner.claimDelivery(route, DeliveryClaimKind.BATCH_ENQUEUE, 7L,
                RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(owner.tryStartSend(delivery));
        var entered = new CountDownLatch(1);
        var release = new CountDownLatch(1);
        publisher.tryRegister().submit(() -> {
            entered.countDown();
            try { assertTrue(release.await(5L, TimeUnit.SECONDS)); }
            catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); throw new AssertionError(interrupted); }
            return true;
        });
        try {
            assertTrue(entered.await(5L, TimeUnit.SECONDS));
            RequestProtocolTestSupport.transport(owner, delivery, DeliveryResult.delivered());
            assertTrue(context.snapshot().delivery().acknowledged());
            assertNull(context.snapshot().selectedResponse());
            assertFalse(context.data().getFuture().isDone());
            var registration = owner.tryClaim(reservation, 4L, "victim after confirmation").orElseThrow();
            assertTrue(RequestProtocolTestSupport.preemptionPhase(owner, registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
            var drained = new CompletableFuture<Void>();
            publisher.tryRegister().submit(() -> { drained.complete(null); return true; });
            release.countDown();
            drained.get(5L, TimeUnit.SECONDS); // The sole response worker has executed the exact ACK task.
            when(decode.updatePreemption(4L, DecodeResources.PreemptionUpdate.active(reservation))).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED, false));
            assertTrue(RequestProtocolTestSupport.preemptionPhase(owner, registration, PreemptionCancelPhase.NOT_FOUND_STALE));
            RequestProtocolTestSupport.applyPrefillStatus(owner, context, prefill, RoleType.PREFILL,
                    new PrefillState.PrefillRequestStatus(route, PrefillState.PrefillRequestStatus.Kind.ACTIVE, 0L));
            assertNull(context.snapshot().preemption());
            assertTrue(context.data().getFuture().isDone(), "confirmed ACK must retain its frontend completion owner");
            assertTrue(context.data().getFuture().get(5L, TimeUnit.SECONDS).isSuccess());
            assertEquals(ResponseKind.ACK, context.snapshot().selectedResponse().kind());
        } finally {
            release.countDown();
            RequestProtocolTestSupport.applyPrefillStatus(owner, context, prefill, RoleType.PREFILL,
                    new PrefillState.PrefillRequestStatus(route, PrefillState.PrefillRequestStatus.Kind.COMPLETED, 0L));
            RequestProtocolTestSupport.applyDecodeStatus(owner, context, decode,
                    DecodeResources.DecodeRequestStatus.terminal(reservation, 0L));
            SchedulerTestSupport.awaitRequestWork(owner.runtime);
            SchedulerTestSupport.closeRequestExecutors(owner.runtime);
            owner.closePlacement();
        }
    }
}
