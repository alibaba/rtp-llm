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
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.test.util.ReflectionTestUtils;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.*;

/** A failed physical reconciliation must give back its exact state-only operation token. */
class RequestReconciliationFailureOwnershipTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void transientDecodeReconciliationFailureRetainsFactsAndMatchingEvidenceCanRetry(boolean matchingDecodeTerminal) throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        var context = RequestProtocolTestSupport.context(config, 9721L);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var owner = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
        ReflectionTestUtils.setField(owner.runtime, "timer", mock(ExpirationTimer.class));
        RequestProtocolTestSupport.register(owner, context);
        var reservation = new DecodeResources.ReservationHandle(1L, 9721L, 3L);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var prefill = mock(PrefillEndpoint.class);
        var server = new ServerStatus();
        server.setServerIp("127.0.0.1");
        server.setGrpcPort(8090);
        var route = SchedulingTestConfig.createRoute(context, new Response(), server, null,
                prefill, decode, reservation, System.currentTimeMillis());
        var admission = RequestProtocolTestSupport.beginAdmission(owner, context);
        assertNotNull(admission);
        assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                owner.commitRoute(route, RequestProtocolTestSupport.publication(() -> true)));
        admission.finish();
        var delivery = owner.claimDelivery(route, DeliveryClaimKind.BATCH_ENQUEUE, 7L,
                RequestProtocolTestSupport.handoff(() -> true));
        assertTrue(owner.tryStartSend(delivery));
        RequestProtocolTestSupport.transport(owner, delivery,
                DeliveryResult.uncertain(new IllegalStateException("reply lost")));
        var registration = owner.tryClaim(reservation, 4L, "victim").orElseThrow();
        assertTrue(RequestProtocolTestSupport.preemptionPhase(owner, registration, PreemptionCancelPhase.CANCEL_IN_FLIGHT));
        var proof = DecodeResources.PreemptionUpdate.finished(reservation);
        var transientFailure = new IllegalStateException("transient Decode reconciliation failure");
        when(decode.updatePreemption(4L, proof)).thenThrow(transientFailure).thenReturn(DecodeResources.ResourceChange.of(matchingDecodeTerminal
                ? DecodeResources.MutationStatus.STALE : DecodeResources.MutationStatus.APPLIED, false));
        when(SchedulerTestSupport.engineClient(owner).cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenReturn(java.util.concurrent.CompletableFuture.completedFuture(
                        EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        var failed = new PrefillState.PrefillRequestStatus(route, PrefillState.PrefillRequestStatus.Kind.FAILED, 9L);
        try {
            RequestProtocolTestSupport.applyPrefillStatus(owner, context, prefill, RoleType.PREFILL, failed);
            var retained = context.snapshot();
            assertSame(registration, retained.preemption().identity());
            assertNull(retained.preemption().reconciliation(), "exception must return its exact operation token");
            assertEquals(PreemptionCancelPhase.CANCEL_IN_FLIGHT, retained.preemption().phase());
            assertNotNull(retained.preemption().pendingTerminal());
            assertFalse(retained.preemption().finished());
            assertFalse(retained.close().decodeSettled(), "exception cannot fabricate Decode release proof");
            assertNull(retained.outcome());
            assertFalse(context.data().getFuture().isDone());
            if (matchingDecodeTerminal) {
                RequestProtocolTestSupport.applyDecodeStatus(owner, context, decode,
                        DecodeResources.DecodeRequestStatus.terminal(reservation, 27L));
            } else {
                RequestProtocolTestSupport.applyPrefillStatus(owner, context, prefill, RoleType.PREFILL, failed);
            }
            RequestProtocolTestSupport.awaitCondition(() -> context.snapshot().phase() == Phase.CLOSED);
            assertEquals("worker error code 9", context.snapshot().outcome().detail(),
                    "later physical proof must preserve the first worker failure");
            verify(decode, times(matchingDecodeTerminal ? 1 : 2)).updatePreemption(4L, proof);
            assertFalse(context.data().getFuture().get(5L, java.util.concurrent.TimeUnit.SECONDS).isSuccess());
            assertEquals(0L, owner.requests.liveRequestCount());
            SchedulerTestSupport.awaitRequestWork(owner.runtime);
            assertEquals(0L, ((Number) ReflectionTestUtils.getField(owner.runtime, "pendingWork")).longValue());
        } finally {
            // Preserve bounded cleanup even when the pre-fix regression assertion fails.
            var state = context.snapshot();
            if (state.preemption() != null && state.preemption().reconciliation() != null) {
                context.completePreemption(state.preemption().reconciliation(),
                        new PreemptionReport(null, false, false, false, state.preemption().pendingTerminal()),
                        System.currentTimeMillis());
            }
            if (context.snapshot().phase() != Phase.CLOSED) {
                when(decode.updatePreemption(4L, proof)).thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(org.flexlb.balance.endpoint.DecodeResources.MutationStatus.APPLIED, false));
                RequestProtocolTestSupport.applyPrefillStatus(owner, context, prefill, RoleType.PREFILL, failed);
            }
            SchedulerTestSupport.awaitRequestWork(owner.runtime);
            SchedulerTestSupport.closeRequestExecutors(owner.runtime);
            owner.closePlacement();
        }
    }
}
