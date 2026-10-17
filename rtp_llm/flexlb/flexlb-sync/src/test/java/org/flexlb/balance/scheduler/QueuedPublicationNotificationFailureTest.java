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
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/** Real queue replacement cannot revoke the incoming publication after a victim callback fails. */
@Timeout(15)
class QueuedPublicationNotificationFailureTest {
    @ParameterizedTest
    @ValueSource(booleans = {true, false})
    void actualReplacementKeepsIncomingAndSettlesOrRetainsVictim(boolean metricsFail) throws Exception {
        try (var fixture = new Fixture()) {
            var low = fixture.context(9301L, 20);
            var oldRoute = fixture.publish(low);
            assertEquals(1L, fixture.prefill.ownership.progress().queueDepth());
            var failure = new IllegalStateException(metricsFail ? "victim metrics failed" : "victim release failed");
            if (metricsFail) {
                fixture.metricsFailure = failure;
            } else {
                fixture.failedReservation = oldRoute.decodeReservation();
                fixture.releaseFailure = failure;
            }
            var incoming = fixture.context(9302L, 80);
            var selected = fixture.select(incoming);
            var handle = fixture.queue.claimAdmissionHandle(9302L, incoming.data().getFuture());
            assertNotNull(handle);
            try (var plan = RequestProtocolTestSupport.plan(incoming, selected)) {
                var published = fixture.queue.enqueueRoute(plan);
                assertEquals(PlacementResult.Status.SUCCESS, published.status());
                var exact = published.value();
                assertSame(exact, incoming.snapshot().route());
                assertTrue(incoming.snapshot().placement().published());
                assertNull(incoming.snapshot().delivery());
                assertFalse(incoming.data().getFuture().isDone());
                assertTrue(fixture.decode.ownership(exact.decodeReservation()).owned());
                assertEquals(List.of(exact), WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.prefill)).items());
                plan.close();
                assertSame(exact, incoming.snapshot().route());
                assertTrue(fixture.decode.ownership(exact.decodeReservation()).owned());
                verify(fixture.decode, never()).release(eq(exact.decodeReservation()),
                        eq(DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
            } finally { handle.finish(); }
            assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                    low.data().getFuture().get(2L, TimeUnit.SECONDS).getCode());
            assertEquals(1L, fixture.prefill.ownership.progress().queueDepth());
            assertFalse(fixture.prefill.control(oldRoute, org.flexlb.balance.scheduler.WorkerBatcher.Control.REMOVE));
            assertNotNull(SchedulerTestSupport.failure(fixture.queue));
            if (metricsFail) {
                assertEquals(ContextResult.Phase.CLOSED, low.snapshot().phase());
                assertFalse(fixture.decode.ownership(oldRoute.decodeReservation()).owned());
                assertNull(fixture.queue.requests.findActive(9301L));
            } else {
                assertEquals(ContextResult.Phase.ENDING, low.snapshot().phase());
                assertFalse(low.snapshot().close().decodeSettled());
                assertTrue(fixture.decode.ownership(oldRoute.decodeReservation()).owned());
                assertSame(low, fixture.queue.requests.findActive(9301L));
                fixture.releaseFailure = null;
                fixture.queue.requestCleanup(low);
                assertEquals(ContextResult.Phase.CLOSED, low.snapshot().phase());
                assertFalse(fixture.decode.ownership(oldRoute.decodeReservation()).owned());
            }
            assertTrue(incoming.snapshot().preparedDelivery());
            assertTrue(fixture.decode.ownership(incoming.snapshot().route().decodeReservation()).owned());
            assertEquals(1, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, fixture.decode.resourceSnapshot().confirmedCount());
        }
    }

    @Test
    void firstVictimFailureDoesNotSkipSecondPhysicallyReplacedVictim() throws Exception {
        try (var fixture = new Fixture(2)) {
            var first = fixture.context(9321L, 20);
            var firstRoute = fixture.publish(first);
            var second = fixture.context(9322L, 30);
            var secondRoute = fixture.publish(second);
            var observation = new WorkerStatusResponse();
            observation.setRole(RoleType.PREFILL);
            observation.setRunningQueryLen(1L);
            fixture.prefill.publishStatus(fixture.prefill.applyStatusHeartbeat(fixture.prefill.getStatus(),
                    WorkerStatusTestSupport.freezeStatusResponse(fixture.prefill.getStatus(), observation)));
            assertEquals(3L, fixture.prefill.ownership.admissionSummary(80, fixture.prefill.requestLimit).occupiedRequests());
            fixture.metricsFailure = new IllegalStateException("first victim metrics failed");
            var incoming = fixture.context(9323L, 80);
            var actual = fixture.publish(incoming);
            assertEquals(List.of(actual), WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.prefill)).items());
            for (var victim : List.of(first, second)) {
                assertEquals(StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode(),
                        victim.data().getFuture().get(2L, TimeUnit.SECONDS).getCode());
                assertEquals(ContextResult.Phase.CLOSED, victim.snapshot().phase());
            }
            verify(fixture.reporter).reportVictim(30, 80, "prefill_queued", "prefill_inflight_requests");
            assertFalse(fixture.decode.ownership(firstRoute.decodeReservation()).owned());
            assertFalse(fixture.decode.ownership(secondRoute.decodeReservation()).owned());
            assertTrue(fixture.decode.ownership(actual.decodeReservation()).owned());
            assertTrue(incoming.snapshot().preparedDelivery());
            assertNull(incoming.snapshot().delivery());
            assertEquals(1, fixture.decode.resourceSnapshot().queuedCount());
            assertEquals(0, fixture.decode.resourceSnapshot().activeDispatchPermits());
            assertEquals(0, fixture.decode.resourceSnapshot().confirmedCount());
            assertNotNull(SchedulerTestSupport.failure(fixture.queue));
        }
    }

    @Test
    void physicalPublicationFailureStillThrowsAndRollsBackExactDecode() throws Exception {
        try (var fixture = new Fixture()) {
            var low = fixture.context(9311L, 20);
            var oldRoute = fixture.publish(low);
            var incoming = fixture.context(9312L, 80);
            var handle = fixture.queue.claimAdmissionHandle(9312L, incoming.data().getFuture());
            assertNotNull(handle);
            var failure = new IllegalStateException("physical offer failed before enqueue");
            fixture.physicalFailure = failure;
            try (var plan = RequestProtocolTestSupport.plan(incoming, fixture.select(incoming))) {
                assertSame(failure, assertThrows(IllegalStateException.class, () -> fixture.queue.enqueueRoute(plan)));
                var acquired = incoming.snapshot().route();
                assertNotNull(acquired);
                assertFalse(incoming.snapshot().placement().published());
                assertTrue(fixture.decode.ownership(acquired.decodeReservation()).owned());
                plan.close();
                assertNull(incoming.snapshot().route());
                assertFalse(fixture.decode.ownership(acquired.decodeReservation()).owned());
                assertEquals(List.of(oldRoute), WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.prefill)).items());
                assertTrue(fixture.decode.ownership(oldRoute.decodeReservation()).owned());
            } finally {
                handle.finish();
                fixture.physicalFailure = null;
            }
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

        Fixture() { this(1); }

        Fixture(int limit) {
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(limit);
            SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
            SchedulingTestConfig.useFixedWindowDecision(config).setMaxCollectionWaitMs(60_000L);
            config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(10L);
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            queue = (QueuedRequestScheduler) SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class),
                    reporter, mock(RecentCacheKeyTraceReporter.class));
            prefill = spy(EndpointTestSupport.realPrefill(WorkerStatus.createDiscovered(RoleType.PREFILL, null,
                    "127.0.0.1", 18100, 18101, null), config, SchedulerTestSupport.dispatcher(queue.runtime),
                    queue.requests, mock(DeliveryMetricsReporter.class)));
            decode = spy(EndpointTestSupport.decode(WorkerStatus.createDiscovered(RoleType.DECODE, null,
                    "127.0.0.1", 18200, 18201, null), queue.requests));
            doAnswer(call -> {
                assertFalse(WorkerBatcherTestSupport.state(EndpointTestSupport.batcher(prefill)).lock.isHeldByCurrentThread());
                if (metricsFailure != null && call.getArgument(0, Integer.class) == 20) { throw metricsFailure; }
                return null;
            }).when(reporter).reportVictim(anyInt(), anyInt(), anyString(), anyString());
            doAnswer(call -> {
                var exact = queue.requests.findActive(call.getArgument(0, DecodeResources.ReservationHandle.class).requestId());
                assertFalse(exact != null && Thread.holdsLock(exact));
                if (releaseFailure != null && java.util.Objects.equals(failedReservation, call.getArgument(0))) { throw releaseFailure; }
                return call.callRealMethod();
            }).when(decode).release(any(), any());
            doAnswer(call -> {
                if (physicalFailure != null) { throw physicalFailure; }
                return call.callRealMethod();
            }).when(prefill).offerPinned(any(), any(), any());
        }

        RequestContext context(long id, int priority) {
            var context = RequestProtocolTestSupport.context(config, id);
            context.data().setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
            RequestProtocolTestSupport.register(queue, context);
            return context;
        }

        RequestRoute select(RequestContext context) {
            var p = metadata(context, RoleType.PREFILL, prefill);
            var d = metadata(context, RoleType.DECODE, decode);
            return SchedulingTestConfig.prepareRoute(context, route -> {
                route.add(prefill.tryPinGeneration(), p, 1L, prefill.ownership.progress().ownershipVersion());
                route.add(decode.tryPinGeneration(), d, -1L, decode.placementVersion());
                return org.flexlb.balance.PlacementResult.success(route);
            }).value();
        }

        RequestRoute publish(RequestContext context) {
            var handle = queue.claimAdmissionHandle(context.data().getRequestId(), context.data().getFuture());
            try (var plan = RequestProtocolTestSupport.plan(context, select(context))) {
                assertNotNull(handle);
                var publication = queue.enqueueRoute(plan);
                assertEquals(PlacementResult.Status.SUCCESS, publication.status());
                return publication.value();
            } finally { if (handle != null) { handle.finish(); } }
        }

        private ServerStatus metadata(RequestContext context, RoleType role, org.flexlb.balance.endpoint.WorkerEndpoint endpoint) {
            var value = new ServerStatus();
            value.setSuccess(true); value.setRequestId(context.data().getRequestId()); value.setRole(role);
            value.setServerIp(endpoint.getIp()); value.setHttpPort(endpoint.getHttpPort()); value.setGrpcPort(endpoint.getGrpcPort());
            return value;
        }

        @Override public void close() {
            metricsFailure = releaseFailure = physicalFailure = null;
            prefill.close();
            if (RequestProtocolTestSupport.closeAdmissionAndAwaitMutations(queue)) { queue.closeOutstandingAndTerminalize(); }
            for (var context : queue.requests.snapshotActive()) { queue.requestCleanup(context); }
            SchedulerTestSupport.timer(queue.runtime).close();
            SchedulerTestSupport.closeRequestExecutors(queue.runtime);
        }
    }
}
