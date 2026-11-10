package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.TheoryCacheHitReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Waiting ownership is independent of delivery slots; failed publication rolls back only its owner.
 */
@Timeout(15)
class QueuedPublicationNotificationFailureTest {
    @Test
    void cancellationAndArchiveDuringNotificationCannotReclaimTransferredResources()
            throws Exception {
        try (var fixture = new Fixture()) {
            var context = fixture.context(9342L, 50);
            var lifecycle = SchedulerTestServices.lifecycle(fixture.queue);
            var failure = new IllegalStateException("notification failed after cancellation");
            doAnswer(
                            call -> {
                                assertNull(context.snapshot().routingContext());
                                assertNotNull(context.snapshot().route());
                                lifecycle.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
                                context.data()
                                        .getFuture()
                                        .get(2, java.util.concurrent.TimeUnit.SECONDS);
                                SchedulerTestSupport.awaitRequestWork(
                                        SchedulerTestSupport.runtime(lifecycle));
                                assertNull(context.snapshot().route());
                                assertEquals(0, lifecycle.repository.liveRequestCount());
                                throw failure;
                            })
                    .when(fixture.prefill)
                    .signal(PrefillEndpoint.Signal.INPUT);
            assertSame(
                    failure,
                    assertThrows(
                            IllegalStateException.class,
                            () ->
                                    ((QueuedRequestScheduler) lifecycle)
                                            .selectAndPublish(
                                                    context,
                                                    () ->
                                                            RoutingResult.success(
                                                                    fixture.select(context)))));

            verify(fixture.decode, org.mockito.Mockito.never())
                    .release(any(), eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
            verify(fixture.decode).release(any(), any());
            assertEquals(0, fixture.decode.resourceSnapshot().reservedCount());
            assertEquals(0L, fixture.prefill.queueDepth());
        }
    }

    @Test
    void vitFailureReleasesOtherRolesAndOldCallbacksCannotReleaseReplacementRoute()
            throws Exception {
        try (var fixture = new Fixture()) {
            fixture.encoder =
                    new org.flexlb.balance.endpoint.EncoderEndpoint(
                            WorkerStatus.createDiscovered(
                                    RoleType.ENCODER, null, "127.0.0.1", 18300, 18301, null),
                            new CapacityNotifier());
            fixture.vit =
                    new org.flexlb.balance.endpoint.WorkerEndpoint(
                            WorkerStatus.createDiscovered(
                                    RoleType.VIT, null, "127.0.0.1", 18400, 18401, null));
            var context = fixture.context(9403L, 80);
            var oldRoute = fixture.publish(context);
            assertTrue(fixture.encoder.commit(oldRoute, 10, 16L));
            var requeues = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(
                            call -> {
                                assertNull(context.snapshot().route());
                                assertEquals(0L, fixture.encoder.requestCount());
                                assertEquals(0, fixture.decode.resourceSnapshot().reservedCount());
                                assertEquals(0L, fixture.prefill.queueDepth());
                                requeues.incrementAndGet();
                                return true;
                            })
                    .when(fixture.queue)
                    .enqueue(any());
            fixture.vit.getStatus().markRemovedFromDiscovery();
            var lifecycle = SchedulerTestServices.lifecycle(fixture.queue);
            assertTrue(
                    lifecycle.onRouteFailure(
                            oldRoute,
                            ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                            "VIT unavailable"));
            assertEquals(1, requeues.get());
            var reported = org.mockito.ArgumentCaptor.forClass(ContextResult.RouteFailure.class);
            verify(fixture.queue).shouldRetryRouteFailure(reported.capture());
            assertSame(fixture.vit, reported.getValue().failedEndpoint());
            assertEquals(RoleType.VIT, reported.getValue().failedEndpoint().getStatus().getRole());
            fixture.vit =
                    new org.flexlb.balance.endpoint.WorkerEndpoint(
                            WorkerStatus.createDiscovered(
                                    RoleType.VIT, null, "127.0.0.1", 18400, 18401, null));
            var replacement = fixture.publish(context);
            assertTrue(fixture.encoder.commit(replacement, 10, 16L));
            lifecycle.onEndpointRetired(
                    context,
                    new ContextResult.EndpointRetirement(
                            oldRoute,
                            oldRoute.decodeReservation(),
                            RoleType.DECODE,
                            fixture.decode,
                            "Late old retirement",
                            System.currentTimeMillis()));
            assertFalse(
                    lifecycle.onRouteFailure(
                            oldRoute,
                            ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                            "Late old failure"));
            assertSame(replacement, context.snapshot().route());
            assertTrue(fixture.decode.ownership(replacement.decodeReservation()).owned());
            assertEquals(1L, fixture.encoder.requestCount());
            assertEquals(1L, fixture.prefill.queueDepth());
            assertFalse(context.data().getFuture().isDone());
            assertEquals(1, requeues.get());
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.CsvSource({
        "PREFILL,false",
        "DECODE,false",
        "PREFILL,true",
        "DECODE,true"
    })
    void unavailableRoleWithdrawsWholeQueuedRouteBeforeRequeue(RoleType failedRole, boolean retire)
            throws Exception {
        try (var fixture = new Fixture()) {
            var context = fixture.context(9401L, 80);
            var route = fixture.publish(context);
            long sequence = context.data().getWorkerEnqueueSequence();
            long expiresAt = route.expiresAtMs();
            var requeues = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(
                            call -> {
                                assertSame(context, call.getArgument(0));
                                assertNull(context.snapshot().route());
                                assertNull(context.snapshot().routingContext());
                                assertEquals(0L, fixture.prefill.queueDepth());
                                assertEquals(
                                        0L,
                                        fixture.prefill.admissionSummary(0, 0).occupiedRequests());
                                assertEquals(0, fixture.decode.resourceSnapshot().reservedCount());
                                assertEquals(
                                        0,
                                        fixture.decode.resourceSnapshot().activeDispatchPermits());
                                assertEquals(sequence, context.data().getWorkerEnqueueSequence());
                                assertEquals(expiresAt, context.data().getRequestExpiresAtMs());
                                assertFalse(context.data().getFuture().isDone());
                                requeues.incrementAndGet();
                                return true;
                            })
                    .when(fixture.queue)
                    .enqueue(any());
            var failed = failedRole == RoleType.PREFILL ? fixture.prefill : fixture.decode;
            if (retire) failed.close();
            else {
                failed.getStatus().markRemovedFromDiscovery();
                fixture.queue.onEndpointChanged(failed);
                fixture.queue.start();
                RequestProtocolTestSupport.awaitCondition(() -> requeues.get() == 1);
            }
            assertEquals(1, requeues.get());
            assertFalse(
                    SchedulerTestServices.lifecycle(fixture.queue)
                            .onRouteFailure(
                                    route,
                                    ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                                    "Duplicate old notification"));
            assertEquals(1, requeues.get());
            assertFalse(context.data().getFuture().isDone());
        }
    }

    @Test
    void wholeRouteCleanupFailureEndsRequestWithoutRequeue() throws Exception {
        try (var fixture = new Fixture()) {
            var context = fixture.context(9402L, 80);
            var route = fixture.publish(context);
            fixture.failedReservation = route.decodeReservation();
            fixture.releaseFailure = new IllegalStateException("Decode release failed");
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            SchedulerTestServices.lifecycle(fixture.queue)
                                    .onRouteFailure(
                                            route,
                                            ContextResult.RouteFailureReason.ENDPOINT_UNAVAILABLE,
                                            "Worker unavailable"));
            verify(fixture.queue, never()).enqueue(any());
            assertEquals(0L, fixture.prefill.queueDepth());
            assertFalse(
                    context.data()
                            .getFuture()
                            .get(2, java.util.concurrent.TimeUnit.SECONDS)
                            .isSuccess());
        }
    }

    @Test
    void waitingOwnersAreNotVictimsOfDeliveryCapacity() throws Exception {
        try (var fixture = new Fixture(1)) {
            var first = fixture.context(9321L, 20);
            var firstRoute = fixture.publish(first);
            var second = fixture.context(9322L, 30);
            var secondRoute = fixture.publish(second);
            var observation = new WorkerStatusResponse();
            observation.setRole(RoleType.PREFILL);
            observation.setRunningQueryLen(1L);
            fixture.prefill.publishStatus(
                    fixture.prefill.applyStatusHeartbeat(
                            fixture.prefill.getStatus(),
                            WorkerStatusTestSupport.freezeStatusResponse(
                                    fixture.prefill.getStatus(), observation)));
            assertFalse(fixture.prefill.requestCapacityAvailable(fixture.prefill.requestLimit, 1));
            fixture.metricsFailure = new IllegalStateException("unused victim metric");
            var incoming = fixture.context(9323L, 80);
            var actual = fixture.publish(incoming);

            var items = fixture.prefill.captureQueue(Integer.MAX_VALUE).items();
            assertEquals(3, items.size());
            assertTrue(items.containsAll(List.of(firstRoute, secondRoute, actual)));
            for (var owner : List.of(first, second, incoming)) {
                assertFalse(owner.data().getFuture().isDone());
                assertTrue(owner.snapshot().preparedDelivery());
                assertNull(owner.snapshot().delivery());
                assertTrue(
                        fixture.decode
                                .ownership(owner.snapshot().route().decodeReservation())
                                .owned());
            }
            verify(fixture.reporter, never())
                    .reportPreemptedRequest(anyInt(), anyInt(), anyString(), anyString());
            assertEquals(3, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
        }
    }

    @Test
    void physicalPublicationFailureStillThrowsAndRollsBackExactDecode() throws Exception {
        try (var fixture = new Fixture()) {
            var low = fixture.context(9311L, 20);
            var oldRoute = fixture.publish(low);
            var incoming = fixture.context(9312L, 80);
            var handle =
                    SchedulerTestSupport.claim(
                            SchedulerTestServices.lifecycle(fixture.queue),
                            "9312",
                            incoming.data().getFuture());
            assertNotNull(handle);
            var failure = new IllegalStateException("physical offer failed before enqueue");
            fixture.physicalFailure = failure;
            var acquired = fixture.select(incoming);
            try (var plan = RequestProtocolTestSupport.plan(incoming, acquired)) {
                assertSame(
                        failure,
                        assertThrows(
                                IllegalStateException.class,
                                () -> SchedulerTestSupport.enqueueRoute(fixture.queue, plan)));
                assertNull(incoming.snapshot().route());
                assertTrue(fixture.decode.ownership(acquired.decodeReservation()).owned());
                plan.close();
                assertNull(incoming.snapshot().route());
                assertFalse(fixture.decode.ownership(acquired.decodeReservation()).owned());
                assertEquals(
                        List.of(oldRoute), fixture.prefill.captureQueue(Integer.MAX_VALUE).items());
                assertTrue(fixture.decode.ownership(oldRoute.decodeReservation()).owned());
            } finally {
                AbstractRequestScheduler.finishRouting(handle);
                fixture.physicalFailure = null;
            }
        }
    }

    @Test
    void notificationFailureAfterPhysicalEnqueueKeepsTheExactResourceHandoff() throws Exception {
        try (var fixture = new Fixture()) {
            var context = fixture.context(9341L, 50);
            var lifecycle = SchedulerTestServices.lifecycle(fixture.queue);
            var admission =
                    SchedulerTestSupport.claim(lifecycle, "9341", context.data().getFuture());
            var failure = new IllegalStateException("placement notification failed");
            org.mockito.Mockito.doThrow(failure)
                    .when(fixture.prefill)
                    .signal(PrefillEndpoint.Signal.INPUT);
            RouteResult route = fixture.select(context);
            try (var plan = RequestProtocolTestSupport.plan(context, route)) {
                assertSame(
                        failure,
                        assertThrows(
                                IllegalStateException.class,
                                () -> SchedulerTestSupport.enqueueRoute(fixture.queue, plan)));
                assertNotNull(route.prefillReservation());
                assertSame(route, context.snapshot().route());
                plan.close();
                assertSame(route, context.snapshot().route());
                assertTrue(fixture.decode.ownership(route.decodeReservation()).owned());
                assertEquals(List.of(route), fixture.prefill.captureQueue(2).items());
            } finally {
                AbstractRequestScheduler.finishRouting(admission);
            }
            lifecycle.cancel(context, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(
                    context.data()
                            .getFuture()
                            .get(2, java.util.concurrent.TimeUnit.SECONDS)
                            .isSuccess());
            assertFalse(fixture.decode.ownership(route.decodeReservation()).owned());
            assertEquals(0, fixture.prefill.queueDepth());
        }
    }

    /**
     * Injects physical publication, metrics, and release failures into a real queued route lifecycle.
     */
    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final RequestSchedulerReporter reporter = mock(RequestSchedulerReporter.class);
        final QueuedRequestScheduler queue;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;

        volatile RuntimeException metricsFailure;
        volatile RuntimeException releaseFailure;
        volatile RuntimeException physicalFailure;
        volatile DecodeResources.RequestReservation failedReservation;
        org.flexlb.balance.endpoint.EncoderEndpoint encoder;
        org.flexlb.balance.endpoint.WorkerEndpoint vit;

        Fixture() {
            this(1);
        }

        Fixture(int limit) {
            SchedulingTestConfig.useNonBatchDispatcher(config)
                    .setMaxInflightPerPrefillWorker(limit);
            SchedulingTestConfig.allowVictim(config, PreemptionStage.PREFILL_QUEUED);
            SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000L);
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(10L);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            queue =
                    (QueuedRequestScheduler)
                            SchedulerTestServices.scheduler(
                                    SchedulerTestSupport.create(
                                            service,
                                            mock(DeliveryMetricsReporter.class),
                                            reporter,
                                            mock(TheoryCacheHitReporter.class)));
            prefill =
                    spy(
                            EndpointTestSupport.realPrefill(
                                    WorkerStatus.createDiscovered(
                                            RoleType.PREFILL,
                                            null,
                                            "127.0.0.1",
                                            18100,
                                            18101,
                                            null),
                                    config,
                                    SchedulerTestSupport.dispatcher(
                                            org.flexlb.balance.scheduler.SchedulerTestSupport
                                                    .runtime(
                                                            SchedulerTestServices.lifecycle(
                                                                    queue))),
                                    SchedulerTestServices.lifecycle(queue).repository,
                                    mock(DeliveryMetricsReporter.class)));
            decode =
                    spy(
                            EndpointTestSupport.decode(
                                    WorkerStatus.createDiscovered(
                                            RoleType.DECODE, null, "127.0.0.1", 18200, 18201, null),
                                    SchedulerTestServices.lifecycle(queue).repository));
            doAnswer(
                            call -> {
                                org.junit.jupiter.api.Assertions.assertFalse(
                                        org.flexlb.balance.endpoint.EndpointTestSupport.stateLock(
                                                        prefill)
                                                .isHeldByCurrentThread());
                                if (metricsFailure != null
                                        && call.getArgument(0, Integer.class) == 20) {
                                    throw metricsFailure;
                                }
                                return null;
                            })
                    .when(reporter)
                    .reportPreemptedRequest(anyInt(), anyInt(), anyString(), anyString());
            doAnswer(
                            call -> {
                                var exact =
                                        SchedulerTestServices.lifecycle(queue)
                                                .repository
                                                .findActive(
                                                        call.getArgument(
                                                                        0,
                                                                        DecodeResources
                                                                                .RequestReservation
                                                                                .class)
                                                                .requestId());
                                assertFalse(exact != null && Thread.holdsLock(exact));
                                if (releaseFailure != null
                                        && java.util.Objects.equals(
                                                failedReservation, call.getArgument(0))) {
                                    throw releaseFailure;
                                }
                                return call.callRealMethod();
                            })
                    .when(decode)
                    .release(any(), any());
            doAnswer(
                            call -> {
                                if (physicalFailure != null) {
                                    throw physicalFailure;
                                }
                                return call.callRealMethod();
                            })
                    .when(prefill)
                    .enqueueRoute(any(), any());
        }

        @Override
        public void close() {
            metricsFailure = releaseFailure = physicalFailure = null;
            queue.stop();
            prefill.close();
            if (RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(
                    SchedulerTestServices.lifecycle(queue))) {
                SchedulerTestServices.lifecycle(queue).closeOutstandingAndTerminalize();
            }
            for (var context : SchedulerTestServices.lifecycle(queue).repository.snapshotActive()) {
                SchedulerTestServices.lifecycle(queue).scheduleClose(context);
            }
            SchedulerTestSupport.timer(
                            org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                                    SchedulerTestServices.lifecycle(queue)))
                    .close();
            SchedulerTestSupport.closeRequestExecutors(
                    org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(
                            SchedulerTestServices.lifecycle(queue)));
        }

        RequestContext context(long id, int priority) {
            var context = RequestProtocolTestSupport.context(config, id);
            context.data()
                    .setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
            RequestProtocolTestSupport.submitForManualPlacement(queue, context);
            return context;
        }

        RouteResult publish(RequestContext context) {
            var handle =
                    SchedulerTestSupport.claim(
                            SchedulerTestServices.lifecycle(queue),
                            context.data().getRequestId(),
                            context.data().getFuture());
            try (var plan = RequestProtocolTestSupport.plan(context, select(context))) {
                assertNotNull(handle);
                var publication = SchedulerTestSupport.enqueueRoute(queue, plan);
                assertEquals(
                        RoutingResult.Status.SUCCESS, publication.status(), publication::toString);
                return publication.value();
            } finally {
                if (handle != null) {
                    AbstractRequestScheduler.finishRouting(handle);
                }
            }
        }

        RouteResult select(RequestContext context) {
            var p = metadata(context, RoleType.PREFILL, prefill);
            var d = metadata(context, RoleType.DECODE, decode);
            var selectedRoles = new java.util.ArrayList<RoleSelection>();

            selectedRoles.add(SelectionTestSupport.metadata(prefill, p, 1L));
            selectedRoles.add(SelectionTestSupport.metadata(decode, d, -1L));
            if (encoder != null)
                selectedRoles.add(
                        SelectionTestSupport.metadata(
                                encoder, metadata(context, RoleType.ENCODER, encoder), 0L));
            if (vit != null)
                selectedRoles.add(
                        SelectionTestSupport.metadata(
                                vit, metadata(context, RoleType.VIT, vit), 0L));

            return SchedulerTestServices.registerSelection(context, selectedRoles).value();
        }

        private ServerStatus metadata(
                RequestContext context,
                RoleType role,
                org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {
            var value = new ServerStatus();
            value.setSuccess(true);
            value.setRequestId(context.data().getRequestId());
            value.setRole(role);
            value.setServerIp(endpoint.getIp());
            value.setHttpPort(endpoint.getHttpPort());
            value.setGrpcPort(endpoint.getGrpcPort());
            return value;
        }
    }
}
