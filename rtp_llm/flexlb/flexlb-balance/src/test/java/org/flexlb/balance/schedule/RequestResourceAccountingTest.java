package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.metrics.RequestSchedulerReporter;
import org.flexlb.balance.requestlog.TheoryCacheHitReporter;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.DecodeResources.Capacity;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.EndpointTestSupport;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.rpc.EngineGrpcClient;
import org.flexlb.worker.status.TaskInfo;
import org.flexlb.worker.status.TaskPhase;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.flexlb.worker.status.WorkerStatusTestSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Resource assertions use real endpoint ledgers, not invocations of mocked release methods.
 */
class RequestResourceAccountingTest {
    private static final long ID = 101L;
    private static final long HARD_KV = 160L;
    private static final long EXPECTED_KV = 320L;
    private static final long TOTAL_KV = 10_000L;
    private static final long TTL = TimeUnit.HOURS.toMillis(1);

    @Test
    @org.junit.jupiter.api.Timeout(10)
    void cancellationReturnsBeforeEngineReplyAndRetainsCapacityUntilCleanup() throws Exception {
        try (Fixture f = new Fixture()) {
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            assertNotNull(f.requests.cancel(String.valueOf(ID), CancelReason.CLIENT_CANCELLED));
            assertFalse(f.cleanup.isDone());
            assertTrue(f.cancelSent.await(5, TimeUnit.SECONDS));
            assertEquals(RequestContext.Phase.ENDING, f.item.ctx().snapshot().phase());
            assertEquals(1, f.decode.selectionSnapshot().usage().occupiedRequests());
            f.proveCleanup();
            assertEquals(RequestContext.Phase.CLOSED, f.item.ctx().snapshot().phase());
            assertFalse(f.decode.resourceSnapshot().requests().containsKey(String.valueOf(ID)));
            f.decodeStatus(Map.of(), Map.of(), TOTAL_KV);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void engineReplyAfterWorkerCleanupDoesNotChangeArchivedRequest() throws Exception {
        try (Fixture f = new Fixture()) {
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            f.requests.cancel(String.valueOf(ID), CancelReason.CLIENT_CANCELLED);
            assertTrue(f.cancelSent.await(5, TimeUnit.SECONDS));
            f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            f.assertEmpty();
            var archived = f.requests.repository.getRequestState(String.valueOf(ID));
            assertFalse(f.requests.isCurrentRequest(f.item.ctx()));
            f.proveCleanup();
            assertSame(archived, f.requests.repository.getRequestState(String.valueOf(ID)));
            f.assertCapacityReusable();
        }
    }

    @Test
    void partialCleanupFailureStillAttemptsOtherResourcesAndPreservesDecodeSettlement() {
        var context = RequestProtocolTestSupport.context(SchedulingTestConfig.newConfig(), ID);
        var prefill = mock(PrefillEndpoint.class);
        var decode = RequestProtocolTestSupport.decodeEndpoint();
        var reservation = new DecodeEndpoint.RequestRegistration(1L, String.valueOf(ID));
        var item =
                SchedulingTestConfig.createRoute(
                        context,
                        new Response(),
                        null,
                        null,
                        prefill,
                        decode,
                        reservation,
                        System.currentTimeMillis());
        when(decode.releaseRequest(reservation)).thenReturn(true);
        var releaseFailure = new IllegalStateException("Prefill release failed");
        when(prefill.releaseRequest(item)).thenThrow(releaseFailure);
        RequestProtocolTestSupport.closing(item);
        var firstFailure =
                AbstractRequestScheduler.releaseRouteResources(
                        item, WorkerEndpoint.ReleaseReason.NOT_SENT, false);
        assertFalse(context.snapshot().prefillCleanupCompleted());
        assertTrue(context.snapshot().decodeCleanupCompleted());
        assertSame(releaseFailure, firstFailure);
        org.mockito.Mockito.verify(decode).releaseRequest(reservation);

        org.mockito.Mockito.doReturn(true).when(prefill).releaseRequest(item);
        assertNull(
                AbstractRequestScheduler.releaseRouteResources(
                        item, WorkerEndpoint.ReleaseReason.NOT_SENT, false));
        assertTrue(context.snapshot().prefillCleanupCompleted());
        assertTrue(context.snapshot().decodeCleanupCompleted());
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.times(1))
                .releaseRequest(reservation);
    }

    @ParameterizedTest
    @EnumSource(LocalEnd.class)
    void localTerminationReclaimsReservationAndPrefillInflight(LocalEnd end) throws Exception {
        try (Fixture f = new Fixture(true)) {
            f.assertReserved();
            switch (end) {
                case CANCEL -> f.requests.cancel(String.valueOf(ID), CancelReason.CLIENT_CANCELLED);
                case FUTURE_CANCEL -> assertTrue(f.item.ctx().getFuture().cancel(false));
                case EXPIRE -> f.expire();
                case SHUTDOWN -> {
                    f.requests.cancel(String.valueOf(ID), CancelReason.SHUTDOWN);
                    f.admission
                            .requestContext()
                            .getRequestScheduler()
                            .finishSelectionWithoutCommit(f.admission, null);
                    assertTrue(
                            RequestProtocolTestSupport.closeRegistrationAndAwaitScheduling(
                                    f.requests));
                    f.requests.closeOutstandingAndTerminalize();
                }
            }
            f.admission
                    .requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(f.admission, null);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @ParameterizedTest
    @EnumSource(CancelReason.class)
    void admissionRetainsResourcesUntilItsOwnerCloses(CancelReason reason) throws Exception {
        try (Fixture f = new Fixture(true)) {
            f.assertReserved();
            if (reason == CancelReason.CLIENT_CANCELLED) {
                f.requests.cancel(String.valueOf(ID), reason);
            } else {
                f.expire();
            }
            assertFalse(f.item.ctx().getFuture().isDone());
            f.assertReserved();
            f.admission
                    .requestContext()
                    .getRequestScheduler()
                    .finishSelectionWithoutCommit(f.admission, null);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    @Test
    void localRollbackCannotReleaseAnEngineOwnerWhoseProjectionHasNotArrived() throws Exception {
        try (Fixture f = new Fixture()) {
            f.decodeStatus(Map.of("101", task(ID)), Map.of(), TOTAL_KV - HARD_KV);
            // DiscoveryEndpoint ownership can advance before its notification reaches the context.
            f.requests.cancel(String.valueOf(ID), CancelReason.CLIENT_CANCELLED);
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(f.requests));
            assertEquals(1, SchedulerTestSupport.repository(f.requests).liveRequestCount());
            assertEquals(RequestContext.Phase.ENDING, f.item.ctx().snapshot().phase());
            assertEquals(0, f.prefill.stats().locallyOwnedRequests());
            assertEquals(0L, f.decode.selectionSnapshot().usage().hardReservedKvTokens());
            assertEquals(1, f.decode.selectionSnapshot().usage().occupiedRequests());
            f.decodeStatus(Map.of(), Map.of("101", task(ID)), TOTAL_KV);
            f.assertEmpty();
            f.assertCapacityReusable();
        }
    }

    private static TaskInfo task(long id) {
        var task = new TaskInfo();
        task.setRequestId(String.valueOf(id));
        task.setInputLength(HARD_KV);
        task.setPhase(TaskPhase.RUNNING);
        return task;
    }

    private static WorkerStatusResponse status(
            RoleType role,
            long version,
            Map<String, TaskInfo> running,
            Map<String, TaskInfo> finished,
            long freeKv) {
        var response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(finished.isEmpty() ? 0L : version);
        response.setRunningTaskInfo(running);
        response.setRunningQueryLen((long) running.size());
        response.setFinishedTaskInfo(finished);
        response.setTotalKvCacheTokens(TOTAL_KV);
        response.setAvailableKvCacheTokens(freeKv);
        return response;
    }

    private static void applyStatus(WorkerEndpoint endpoint, WorkerStatusResponse response) {
        var worker = endpoint.getStatus();
        WorkerEndpoint.StatusChange projection;
        worker.lock.lock();
        try {
            projection =
                    endpoint.applyWorkerStatus(
                            worker,
                            worker.prepareStatus(
                                    WorkerStatusTestSupport.freezeStatusResponse(
                                            worker, response)));
        } finally {
            worker.lock.unlock();
        }
        endpoint.publishStatus(projection);
    }

    private enum LocalEnd {
        CANCEL,
        FUTURE_CANCEL,
        EXPIRE,
        SHUTDOWN
    }

    /**
     * Assembles exact P/D registrations and handoff permits to compare accounting before and after lifecycle events.
     */
    private static final class Fixture implements AutoCloseable {
        final AbstractRequestScheduler requests;
        final SchedulerTestServices runtime;
        final PrefillEndpoint prefill;
        final DecodeEndpoint decode;
        final RequestResult item;
        final DecodeEndpoint.RequestRegistration reservation;
        final PrefillEndpoint.RequestRegistration routeReservation;
        final boolean prefillHandoff;
        final SelectionAttempt admission;
        final DecodeResources.AdmissionCapacity capacity =
                new DecodeResources.AdmissionCapacity(1, 90);

        long version = 1;

        final java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply> cleanup =
                new java.util.concurrent.CompletableFuture<>();
        final java.util.concurrent.CountDownLatch cancelSent =
                new java.util.concurrent.CountDownLatch(1);

        Fixture() {
            this(false);
        }

        Fixture(boolean keepAdmissionOpen) {
            var config = SchedulingTestConfig.newConfig();
            config.setScheduler(ConfigTestSupport.direct());
            config.getRequestLifecycle().getRequest().setTimeoutMs(TTL);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            var reporter = mock(DeliveryMetricsReporter.class);
            var requestReporter = mock(RequestSchedulerReporter.class);
            requests =
                    SchedulerTestSupport.create(
                            service, reporter, requestReporter, mock(TheoryCacheHitReporter.class));
            var channel = mock(EngineGrpcClient.class);
            when(channel.cancelAsync(any(), anyInt(), any(), anyLong()))
                    .thenAnswer(
                            call -> {
                                cancelSent.countDown();
                                return cleanup;
                            });
            org.flexlb.balance.schedule.DeliveryTestSupport.setField(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requests).dispatcher,
                    "grpcClient",
                    channel);
            var projector = requests;
            var endpoints =
                    new EndpointRegistry(
                            service,
                            SchedulerTestSupport.repository(projector),
                            reporter,
                            new CapacityNotifier(),
                            org.mockito.Mockito.mock(
                                    org.flexlb.cache.match.CacheAwareService.class));
            runtime = org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requests);
            org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                    runtime, "endpoints", endpoints);
            var worker =
                    WorkerStatus.createDiscovered(
                            RoleType.PREFILL, "g", "127.0.0.1", 8080, 8081, "test");
            worker.lock.lock();
            try {
                prefill =
                        org.mockito.Mockito.spy(
                                (PrefillEndpoint)
                                        endpoints
                                                .updateEndpoint(
                                                        worker.getLogicalIpPort(),
                                                        worker,
                                                        worker.prepareStatus(
                                                                WorkerStatusTestSupport
                                                                        .freezeStatusResponse(
                                                                                worker,
                                                                                status(
                                                                                        RoleType
                                                                                                .PREFILL,
                                                                                        1,
                                                                                        Map.of(),
                                                                                        Map.of(),
                                                                                        TOTAL_KV))))
                                                .endpoint());
            } finally {
                worker.lock.unlock();
            }
            decode =
                    EndpointTestSupport.decode(
                            WorkerStatus.createDiscovered(
                                    RoleType.DECODE, "g", "127.0.0.2", 8080, 8081, "test"),
                            SchedulerTestSupport.repository(projector));
            decodeStatus(Map.of(), Map.of(), TOTAL_KV);
            {
                var selectedEndpoint = decode;

                reservation =
                        decode.tryRegisterSelection(
                                        decode.selectionSnapshot(),
                                        String.valueOf(ID),
                                        HARD_KV,
                                        EXPECTED_KV,
                                        50,
                                        capacity)
                                .value();
            }
            assertNotNull(reservation);
            var context = RequestProtocolTestSupport.context(config, ID);
            var future = RequestProtocolTestSupport.submitForManualPlacement(requests, context);
            context.setFuture(future);
            item =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            context,
                            prefill,
                            decode,
                            reservation,
                            30_000L,
                            System.currentTimeMillis());
            admission = SchedulerTestSupport.claim(requests, String.valueOf(ID), future);
            assertNotNull(admission);
            routeReservation = item.prefillRegistration();
            context.registerSelectedRoute(admission, item);
            prefillHandoff = !keepAdmissionOpen && prefill.commitSelection(item);
            if (!keepAdmissionOpen) {
                assertTrue(prefillHandoff);
                assertEquals(
                        org.flexlb.balance.strategy.RoutingResult.Status.SUCCESS,
                        RequestProtocolTestSupport.commitRoute(
                                requests,
                                item,
                                RequestProtocolTestSupport.publication(() -> true)));
                admission
                        .requestContext()
                        .getRequestScheduler()
                        .finishSelectionWithoutCommit(admission, null);
            }
        }

        @Override
        public void close() {
            try {
                admission
                        .requestContext()
                        .getRequestScheduler()
                        .finishSelectionWithoutCommit(admission, null);

                prefill.rollbackSelection(routeReservation);
                SchedulerTestSupport.shutdown(runtime);
            } finally {
                org.flexlb.worker.endpoint.EndpointTestSupport.closeAndWait(decode);
            }
        }

        void assertHandedOff() {
            assertReserved();
            assertEquals(1, decode.selectionSnapshot().usage().occupiedRequests());
            assertFalse(
                    capacity.evaluate(
                                    decode.selectionSnapshot().usage(),
                                    HARD_KV,
                                    EXPECTED_KV,
                                    Capacity.NONE)
                            .isZero());
        }

        void assertReserved() {
            assertEquals(1, SchedulerTestSupport.repository(requests).liveRequestCount());
            assertEquals(1, prefill.stats().locallyOwnedRequests());
            assertEquals(HARD_KV, decode.selectionSnapshot().usage().hardReservedKvTokens());
            assertEquals(
                    EXPECTED_KV, EndpointTestSupport.expectedReservedKv(decode.resourceSnapshot()));
        }

        void assertCapacityReusable() {
            {
                var selectedEndpoint = decode;

                var next =
                        decode.tryRegisterSelection(
                                        decode.selectionSnapshot(),
                                        String.valueOf(999L),
                                        HARD_KV,
                                        EXPECTED_KV,
                                        50,
                                        capacity)
                                .value();
                assertNotNull(next);
                decode.releaseRequest(next);
            }
            // A second exact request must be able to use the sole Prefill slot too.
            var nextContext = RequestProtocolTestSupport.context(item.ctx().getConfig(), 999L);
            nextContext.setFuture(new CompletableFuture<>());
            var next =
                    RequestExecutionTestSupport.registerPrefillRoute(
                            nextContext, prefill, null, null, 1L, System.currentTimeMillis());
            {
                var selectedEndpoint = prefill;

                prefill.rollbackSelection(next.prefillRegistration());
            }
            assertEmpty();
        }

        void assertEmpty() {
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requests));
            assertEquals(
                    0,
                    SchedulerTestSupport.repository(requests).liveRequestCount(),
                    "request inflight");

            assertEquals(0, prefill.stats().locallyOwnedRequests(), "Prefill inflight");
            assertEquals(0, prefill.stats().locallyOwnedRequests(), "Prefill individual lease");

            var view = decode.resourceSnapshot();
            assertEquals(
                    0,
                    view.selection().usage().hardReservedKvTokens(),
                    "Decode hard KV reservation");
            assertEquals(
                    0,
                    EndpointTestSupport.expectedReservedKv(view),
                    "Decode expected KV reservation");
            assertEquals(
                    0,
                    view.requests().values().stream()
                            .filter(snapshotRequest -> !snapshotRequest.phase().isEngineConfirmed())
                            .count(),
                    "Decode queued ownership");
            assertEquals(
                    0,
                    view.requests().values().stream()
                            .filter(
                                    snapshotRequest ->
                                            snapshotRequest.phase()
                                                    == org.flexlb.worker.status.DecodeTaskPhase
                                                            .ACCEPTED_NOT_RUNNING)
                            .count(),
                    "Decode accepted streams");
            assertEquals(
                    0,
                    view.requests().values().stream()
                            .filter(
                                    snapshotRequest ->
                                            snapshotRequest.phase()
                                                    == org.flexlb.worker.status.DecodeTaskPhase
                                                            .RUNNING)
                            .count(),
                    "Decode running streams");
            assertEquals(0, view.selection().usage().occupiedRequests(), "Decode request capacity");
            assertTrue(
                    view.requests().values().stream()
                                    .filter(
                                            snapshotRequest ->
                                                    !snapshotRequest.phase().isEngineConfirmed())
                                    .count()
                            == 0);
            assertTrue(
                    view.requests().values().stream()
                                    .filter(
                                            snapshotRequest ->
                                                    snapshotRequest.phase().isEngineConfirmed())
                                    .count()
                            == 0);
        }

        void proveCleanup() throws Exception {
            var closeQueue =
                    (RequestCloseQueue) DeliveryTestSupport.getField(requests, "closeQueue");
            closeQueue.awaitIdle();
            cleanup.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            runtime.dispatcher.awaitStops();
            closeQueue.awaitIdle();
            SchedulerTestSupport.awaitRequestWork(
                    org.flexlb.balance.schedule.SchedulerTestSupport.runtime(requests));
        }

        void expire() {
            RequestProtocolTestSupport.expireInactiveRequest(
                    requests,
                    requests.repository.findActive(String.valueOf(ID)),
                    System.currentTimeMillis() + TTL + 1);
        }

        void decodeStatus(
                Map<String, TaskInfo> running, Map<String, TaskInfo> finished, long freeKv) {
            applyStatus(decode, status(RoleType.DECODE, version++, running, finished, freeKv));
        }
    }
}
