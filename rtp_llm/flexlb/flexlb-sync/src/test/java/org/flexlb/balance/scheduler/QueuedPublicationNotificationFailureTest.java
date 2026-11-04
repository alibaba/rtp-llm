package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointTestSupport;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
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
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.anyInt;
import static org.mockito.Mockito.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Waiting ownership is independent of delivery slots; failed publication rolls back only its owner. */
@Timeout(15)
class QueuedPublicationNotificationFailureTest {
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
                    .reportVictim(anyInt(), anyInt(), anyString(), anyString());
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
                    SchedulerTestServices.lifecycle(fixture.queue)
                            .claimAdmissionHandle(9312L, incoming.data().getFuture());
            assertNotNull(handle);
            var failure = new IllegalStateException("physical offer failed before enqueue");
            fixture.physicalFailure = failure;
            try (var plan = RequestProtocolTestSupport.plan(incoming, fixture.select(incoming))) {
                assertSame(
                        failure,
                        assertThrows(
                                IllegalStateException.class,
                                () -> SchedulerTestSupport.enqueueRoute(fixture.queue, plan)));
                var acquired = incoming.snapshot().route();
                assertNotNull(acquired);
                assertFalse(incoming.snapshot().placement().published());
                assertTrue(fixture.decode.ownership(acquired.decodeReservation()).owned());
                plan.close();
                assertNull(incoming.snapshot().route());
                assertFalse(fixture.decode.ownership(acquired.decodeReservation()).owned());
                assertEquals(
                        List.of(oldRoute), fixture.prefill.captureQueue(Integer.MAX_VALUE).items());
                assertTrue(fixture.decode.ownership(oldRoute.decodeReservation()).owned());
            } finally {
                handle.finish();
                fixture.physicalFailure = null;
            }
        }
    }

    @Test
    void notificationFailureAfterPhysicalEnqueueKeepsTheExactResourceHandoff() throws Exception {
        try (var fixture = new Fixture()) {
            var context = fixture.context(9341L, 50);
            var lifecycle = SchedulerTestServices.lifecycle(fixture.queue);
            var admission = lifecycle.claimAdmissionHandle(9341L, context.data().getFuture());
            var failure = new IllegalStateException("placement notification failed");
            org.mockito.Mockito.doThrow(failure)
                    .when(fixture.prefill)
                    .signal(PrefillEndpoint.Signal.PLACEMENT);
            RequestRoute route = fixture.select(context);
            try (var plan = RequestProtocolTestSupport.plan(context, route)) {
                assertSame(
                        failure,
                        assertThrows(
                                IllegalStateException.class,
                                () -> SchedulerTestSupport.enqueueRoute(fixture.queue, plan)));
                assertNotNull(route.prefillReservation());
                assertTrue(context.snapshot().placement().published());
                plan.close();
                assertSame(route, context.snapshot().route());
                assertTrue(fixture.decode.ownership(route.decodeReservation()).owned());
                assertEquals(List.of(route), fixture.prefill.captureQueue(2).items());
            } finally {
                admission.finish();
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

    private static final class Fixture implements AutoCloseable {
        final FlexlbConfig config = SchedulingTestConfig.newConfig();
        final RequestSchedulerReporter reporter = mock(RequestSchedulerReporter.class);
        final QueuedRequestScheduler queue;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        volatile RuntimeException metricsFailure;
        volatile RuntimeException releaseFailure;
        volatile RuntimeException physicalFailure;
        volatile DecodeResources.ReservationHandle failedReservation;

        Fixture() {
            this(1);
        }

        Fixture(int limit) {
            SchedulingTestConfig.useNonBatchDispatcher(config)
                    .setMaxInflightPerPrefillWorker(limit);
            SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
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
                                            mock(RecentCacheKeyTraceReporter.class)));
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
                                    SchedulerTestServices.lifecycle(queue).requests,
                                    mock(DeliveryMetricsReporter.class)));
            decode =
                    spy(
                            EndpointTestSupport.decode(
                                    WorkerStatus.createDiscovered(
                                            RoleType.DECODE, null, "127.0.0.1", 18200, 18201, null),
                                    SchedulerTestServices.lifecycle(queue).requests));
            doAnswer(
                            call -> {
                                org.junit.jupiter.api.Assertions.assertFalse(
                                        prefill.resourceLock().isHeldByCurrentThread());
                                if (metricsFailure != null
                                        && call.getArgument(0, Integer.class) == 20) {
                                    throw metricsFailure;
                                }
                                return null;
                            })
                    .when(reporter)
                    .reportVictim(anyInt(), anyInt(), anyString(), anyString());
            doAnswer(
                            call -> {
                                var exact =
                                        SchedulerTestServices.lifecycle(queue)
                                                .requests
                                                .findActive(
                                                        call.getArgument(
                                                                        0,
                                                                        DecodeResources
                                                                                .ReservationHandle
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
                    .offerPinned(any(), any(), any());
        }

        RequestContext context(long id, int priority) {
            var context = RequestProtocolTestSupport.context(config, id);
            context.data()
                    .setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
            RequestProtocolTestSupport.register(queue, context);
            return context;
        }

        RequestRoute select(RequestContext context) {
            var p = metadata(context, RoleType.PREFILL, prefill);
            var d = metadata(context, RoleType.DECODE, decode);
            return SchedulingTestConfig.prepareRoute(
                            context,
                            route -> {
                                route.add(
                                        prefill.tryPinGeneration(),
                                        p,
                                        1L,
                                        prefill.progress().ownershipVersion());
                                route.add(
                                        decode.tryPinGeneration(),
                                        d,
                                        -1L,
                                        decode.placementVersion());
                                return org.flexlb.balance.PlacementResult.success(route);
                            })
                    .value();
        }

        RequestRoute publish(RequestContext context) {
            var handle =
                    SchedulerTestServices.lifecycle(queue)
                            .claimAdmissionHandle(
                                    context.data().getRequestId(), context.data().getFuture());
            try (var plan = RequestProtocolTestSupport.plan(context, select(context))) {
                assertNotNull(handle);
                var publication = SchedulerTestSupport.enqueueRoute(queue, plan);
                assertEquals(PlacementResult.Status.SUCCESS, publication.status());
                return publication.value();
            } finally {
                if (handle != null) {
                    handle.finish();
                }
            }
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

        @Override
        public void close() {
            metricsFailure = releaseFailure = physicalFailure = null;
            prefill.close();
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(
                    SchedulerTestServices.lifecycle(queue))) {
                SchedulerTestServices.lifecycle(queue).closeOutstandingAndTerminalize();
            }
            for (var context : SchedulerTestServices.lifecycle(queue).requests.snapshotActive()) {
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
    }
}
