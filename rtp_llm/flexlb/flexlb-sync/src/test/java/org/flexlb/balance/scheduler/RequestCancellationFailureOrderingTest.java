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
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Phase;
import static org.flexlb.balance.scheduler.ContextResult.PreemptionReport;
import static org.flexlb.balance.scheduler.ContextResult.PreparationFailure;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * A later delivery failure cannot change the flow responsible for a previously accepted cancellation.
 */
@Timeout(15)
class RequestCancellationFailureOrderingTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryFailureOnlyBypassesPlacementWithoutAnEarlierCancellation(boolean cancelFirst) {
        var context = new RequestContext(mock(FlexlbConfig.class));
        context.register(
                "9501",
                1000L,
                100_000L,
                10_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var scope = RequestCleanupPassTest.requireValue(context.beginPlacement(null));
        var route = RequestCleanupPassTest.route(context, 9501L);
        org.junit.jupiter.api.Assertions.assertTrue(context.publishRoute(scope, route));
        if (cancelFirst) {
            cancel(context);
        }
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordDeliveryResult(
                                new PreparationFailure(
                                        null, route, "late delivery failure", 1002L)));
        var failure =
                new EndEvidence(
                        route,
                        EndKind.DELIVERY_FAILURE,
                        null,
                        false,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "late delivery failure",
                        false,
                        false);
        var end = context.beginClose(failure, 1003L);
        if (cancelFirst) {
            assertNotNull(end.placement());
            assertEquals(Phase.OPEN, context.snapshot().phase());
            assertNull(context.snapshot().outcome());
            assertEquals(EndKind.FRONTEND, context.snapshot().pendingEnd().kind());
            org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                    context,
                    observedRequest ->
                            observedRequest.completePlacement(
                                    scope, PublicationResult.PUBLISHED, 1004L));
            RequestCleanupPassTest.ended(
                    context.beginClose(context.snapshot().pendingEnd(), 1005L));
            assertFirstCause(context);
        } else {
            RequestCleanupPassTest.ended(end);
            assertEquals(RequestState.Phase.FAILED, context.snapshot().outcome().phase());
            assertNotNull(
                    context.snapshot().placement(),
                    "definite failure still publishes independently of physical placement cleanup");
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void deliveryFailureOnlyBypassesPreemptionWithoutAnEarlierCancellation(boolean cancelFirst) {
        var context = RequestCleanupPassTest.openContext(9502L);
        var route = context.snapshot().route();
        var preemption =
                RequestCleanupPassTest.requireValue(
                        context.beginPreemption(
                                route.decodeReservation(),
                                11L,
                                new CancelTarget("127.0.0.1", 8090)));
        if (cancelFirst) {
            cancel(context);
        }
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordDeliveryResult(
                                new PreparationFailure(
                                        null, route, "late delivery failure", 1002L)));
        var failure =
                new EndEvidence(
                        route,
                        EndKind.DELIVERY_FAILURE,
                        null,
                        false,
                        0L,
                        StrategyErrorType.DISPATCH_FAILED,
                        "late delivery failure",
                        false,
                        false);
        var end = context.beginClose(failure, 1003L);
        if (cancelFirst) {
            assertNotNull(end.preemption());
            assertEquals(Phase.ENDING, context.snapshot().phase());
            assertFirstCause(context);
            assertEquals(
                    EndKind.DELIVERY_FAILURE,
                    context.snapshot().preemption().pendingTerminal().kind());
            assertSame(preemption, context.snapshot().preemption().identity());
            context.completePreemption(
                    preemption, new PreemptionReport(null, false, false, true, null), 1004L);
            assertNull(
                    context.snapshot().preemption(),
                    "the exact locally releasable owner ends its attempt");
            RequestCleanupPassTest.ended(
                    context.beginClose(context.snapshot().pendingEnd(), 1005L));
            assertFirstCause(context);
        } else {
            RequestCleanupPassTest.ended(end);
            assertEquals(RequestState.Phase.FAILED, context.snapshot().outcome().phase());
            assertSame(preemption, context.snapshot().preemption().identity());
            assertFalse(
                    context.snapshot().preemption().finished(),
                    "publishing failure cannot fabricate Decode release proof");
        }
    }

    @Test
    void
            cancelledAdmissionCannotReserveFrontendPublicationBeforeItsOwnerEndsEvenAfterLateFailureAndActivity()
                    throws Exception {
        var config = SchedulingTestConfig.newConfig();
        config.setScheduler(ConfigTestSupport.direct());
        config.setDispatcher(ConfigTestSupport.nonBatch());
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var owner =
                SchedulerTestSupport.create(
                        service,
                        mock(DeliveryMetricsReporter.class),
                        mock(RequestSchedulerReporter.class),
                        mock(TheoryCacheHitReporter.class));
        var context = RequestProtocolTestSupport.context(config, 9503L);
        var future = RequestProtocolTestSupport.submitForManualPlacement(owner, context);
        var prefill = mock(PrefillEndpoint.class);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeResources.RequestReservation(1L, String.valueOf(9503L), 1L);
        var route =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        decode,
                        reservation,
                        context.snapshot().createdAtMs());
        var publications = new AtomicInteger();
        var completions = org.mockito.Mockito.spy(owner.responseCompletions);
        SchedulerTestServices.setDependency(
                SchedulerTestSupport.runtime(owner), "responseCompletions", completions);
        doAnswer(
                        call -> {
                            publications.incrementAndGet();
                            return call.callRealMethod();
                        })
                .when(completions)
                .register();
        try {
            var admission = SchedulerTestSupport.claim(owner, "9503", future);
            {
                assertNotNull(admission);
                try {
                    assertEquals(
                            org.flexlb.balance.PlacementResult.Status.SUCCESS,
                            RequestProtocolTestSupport.commitRoute(
                                    owner,
                                    route,
                                    RequestProtocolTestSupport.publication(() -> true)));
                    owner.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
                    owner.onDeliveryResult(
                            new PreparationFailure(
                                    null, route, "late failure", System.currentTimeMillis()));
                    assertEquals(
                            0,
                            publications.get(),
                            "no accepted response task can race an instantaneous isDone assertion");
                    assertFalse(future.isDone());
                    assertNull(context.snapshot().outcome());
                    var inactivity = context.snapshot().deadlines().inactivity();
                    owner.onDeadline(inactivity, inactivity.atMs());
                    owner.onWorkerRequestEvent(
                            context,
                            new WorkerRequestEvent(
                                    route,
                                    reservation,
                                    RoleType.DECODE,
                                    WorkerRequestEventType.ACTIVE,
                                    0L,
                                    null,
                                    inactivity.atMs() + TimeUnit.HOURS.toMillis(1L)));
                    assertTrue(context.snapshot().close().expired());
                    assertNotNull(context.snapshot().placement());
                    assertEquals(EndKind.FRONTEND, context.snapshot().pendingEnd().kind());
                    assertEquals(
                            CancelReason.CLIENT_CANCELLED,
                            context.snapshot().cancellation().reason());
                    assertNull(context.snapshot().outcome());
                    assertEquals(0, publications.get());
                    assertFalse(future.isDone());
                } finally {
                    AbstractRequestScheduler.finishPlacement(admission);
                }
            }
            assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
            RequestProtocolTestSupport.awaitCondition(
                    () -> owner.repository.liveRequestCount() == 0);
            assertEquals(Phase.CLOSED, context.snapshot().phase());
            assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().phase());
            assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
            assertEquals(1, publications.get());
            verify(prefill, times(1)).releaseRequest(route);
            verify(decode, times(1)).release(reservation, DecodeResources.ReleaseReason.EXPIRED);
        } finally {
            SchedulerTestSupport.shutdown(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner));
        }
    }

    private static void cancel(RequestContext context) {
        context.requestCancel(
                new Cancellation(
                        0L,
                        CancelReason.CLIENT_CANCELLED,
                        false,
                        1001L,
                        StrategyErrorType.REQUEST_CANCELLED,
                        "first client cancellation"));
    }

    private static void assertFirstCause(RequestContext context) {
        assertEquals(RequestState.Phase.CANCELLED, context.snapshot().outcome().phase());
        assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().cancellation().reason());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED, context.snapshot().outcome().errorType());
        assertEquals("first client cancellation", context.snapshot().outcome().detail());
    }
}
