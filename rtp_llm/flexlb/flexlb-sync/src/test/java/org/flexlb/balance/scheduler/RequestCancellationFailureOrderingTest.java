package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.scheduler.ContextResult.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** A later delivery failure cannot change the flow responsible for a previously accepted cancellation. */
@Timeout(15)
class RequestCancellationFailureOrderingTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryFailureOnlyBypassesPlacementWithoutAnEarlierCancellation(boolean cancelFirst) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(9501L, 1000L, 100_000L, 10_000L);
        var scope = RequestCleanupPassTest.granted(context.beginPlacement(null));
        var route = RequestCleanupPassTest.route(context, 9501L);
        RequestCleanupPassTest.granted(context.bindPlacement(scope, route, PublicationResult.PREPARED));
        RequestCleanupPassTest.granted(context.bindPlacement(scope, route, PublicationResult.PUBLISHED));
        if (cancelFirst) { cancel(context); }
        context.recordDeliveryResult(new PreparationFailure(null, route, "late delivery failure", 1002L));
        var failure = new EndEvidence(route, EndKind.DELIVERY_FAILURE, null, false, 0L,
                StrategyErrorType.DISPATCH_FAILED, "late delivery failure", false, false);
        var end = context.beginClose(failure, 1003L);
        if (cancelFirst) {
            assertEquals(Blocker.PLACEMENT, assertInstanceOf(Blocked.class, end).blocker());
            assertEquals(Phase.OPEN, context.snapshot().phase());
            assertNull(context.snapshot().outcome());
            assertNull(context.snapshot().close().endOperation());
            assertEquals(EndKind.DELIVERY_FAILURE, context.snapshot().placement().pendingTerminal().kind());
            context.completePlacement(scope, PublicationResult.PUBLISHED, 1004L);
            RequestCleanupPassTest.granted(context.beginClose(context.snapshot().pendingEnd(), 1005L));
            assertFirstCause(context);
        } else {
            RequestCleanupPassTest.granted(end);
            assertEquals(RequestState.Phase.FAILED, context.snapshot().outcome().outcome().phase());
            assertNotNull(context.snapshot().placement(), "definite failure still publishes independently of physical placement cleanup");
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryFailureOnlyBypassesPreemptionWithoutAnEarlierCancellation(boolean cancelFirst) {
        var context = RequestCleanupPassTest.openContext(9502L);
        var route = context.snapshot().route();
        var preemption = RequestCleanupPassTest.granted(context.beginPreemption(route.decodeReservation(),
                11L, new CancelTarget("127.0.0.1", 8090)));
        if (cancelFirst) { cancel(context); }
        context.recordDeliveryResult(new PreparationFailure(null, route, "late delivery failure", 1002L));
        var failure = new EndEvidence(route, EndKind.DELIVERY_FAILURE, null, false, 0L,
                StrategyErrorType.DISPATCH_FAILED, "late delivery failure", false, false);
        var end = context.beginClose(failure, 1003L);
        if (cancelFirst) {
            assertEquals(Blocker.PREEMPTION, assertInstanceOf(Blocked.class, end).blocker());
            assertEquals(Phase.OPEN, context.snapshot().phase());
            assertNull(context.snapshot().outcome());
            assertNull(context.snapshot().close().endOperation());
            assertEquals(EndKind.DELIVERY_FAILURE, context.snapshot().preemption().pendingTerminal().kind());
            assertSame(preemption, context.snapshot().preemption().identity());
            context.completePreemption(preemption, new PreemptionReport(null, false, false, true, null), 1004L);
            assertNull(context.snapshot().preemption(), "the exact locally releasable owner ends its attempt");
            RequestCleanupPassTest.granted(context.beginClose(context.snapshot().pendingEnd(), 1005L));
            assertFirstCause(context);
        } else {
            RequestCleanupPassTest.granted(end);
            assertEquals(RequestState.Phase.FAILED, context.snapshot().outcome().outcome().phase());
            assertSame(preemption, context.snapshot().preemption().identity());
            assertFalse(context.snapshot().preemption().finished(), "publishing failure cannot fabricate Decode release proof");
        }
    }

    @Test
    void cancelledAdmissionCannotReserveFrontendPublicationBeforeItsOwnerEndsEvenAfterLateFailureAndActivity() throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        config.setDispatcher(ConfigTestSupport.nonBatch());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var owner = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
        var context = RequestProtocolTestSupport.context(config, 9503L);
        var future = RequestProtocolTestSupport.register(owner, context);
        var prefill = mock(PrefillEndpoint.class);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeResources.ReservationHandle(1L, 9503L, 1L);
        var route = SchedulingTestConfig.createRoute(context, new Response(), null, null,
                prefill, decode, reservation, context.snapshot().createdAtMs());
        var publications = new AtomicInteger();
        doAnswer(call -> { publications.incrementAndGet(); return call.callRealMethod(); })
                .when(owner.runtime).registerResponseCompletion();
        try {
            try (var admission = owner.claimAdmissionHandle(9503L, future)) {
                assertNotNull(admission);
                try {
                    assertEquals(org.flexlb.balance.PlacementResult.Status.SUCCESS,
                            owner.commitRoute(route, RequestProtocolTestSupport.publication(() -> true)));
                    owner.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
                    owner.onDeliveryEvidence(null, new PreparationFailure(null, route, "late failure", System.currentTimeMillis()));
                    assertEquals(0, publications.get(), "no accepted response task can race an instantaneous isDone assertion");
                    assertFalse(future.isDone());
                    assertNull(context.snapshot().outcome());
                    assertNull(context.snapshot().close().endOperation());
                    var inactivity = context.snapshot().deadlines().inactivity();
                    owner.onDeadline(inactivity, inactivity.atMs());
                    owner.onEngineEvidence(context, new EngineEvidence(route, reservation, RoleType.DECODE, EngineKind.ACTIVE,
                            0L, null, inactivity.atMs() + TimeUnit.HOURS.toMillis(1L)));
                    assertTrue(context.snapshot().close().expired());
                    assertTrue(context.snapshot().placement().inactivityExpired());
                    assertEquals(0, publications.get());
                    assertFalse(future.isDone());
                } finally { admission.finish(); }
            }
            assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(() -> owner.requests.liveRequestCount() == 0);
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().outcome().phase());
            assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
            assertEquals(1, publications.get());
            verify(prefill, times(1)).release(null, route);
            verify(decode, times(1)).release(reservation, DecodeResources.ReleaseReason.EXPIRED);
        } finally { owner.runtime.shutdown(); }
    }

    private static void cancel(RequestContext context) {
        RequestCleanupPassTest.granted(context.requestCancel(new Cancellation(0L, CancelReason.CLIENT_CANCELLED,
                false, 1001L, StrategyErrorType.REQUEST_CANCELLED, "first client cancellation")));
    }

    private static void assertFirstCause(RequestContext context) {
        assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().outcome().phase());
        assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED, context.snapshot().outcome().errorType());
        assertEquals("first client cancellation", context.snapshot().outcome().detail());
    }
}
