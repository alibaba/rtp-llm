package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.DeliveryDispatcher;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.scheduler.WorkerBatcherTestSupport;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** The configured limit, real worker publication lock and canonical request ledger together. */
class WorkerBatcherRequestCapacityTest {

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void waitingRequestsDoNotConsumeDeliverySlotsOrEvictQueuedOwners(boolean priority) {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        if (priority) {
            config.getScheduler().setOrdering(ConfigTestSupport.priority());
        }
        try (Fixture fixture = fixture(config)) {
            List<RequestRoute> routes = new ArrayList<>();
            for (long id = 1; id <= 8; id++) {
                var route = item(config, fixture.endpoint, id, priority ? (int) id : 0);
                routes.add(route);
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, route));
            }
            assertEquals(8, fixture.endpoint.ownership.captureQueue(20).items().size());
            assertTrue(fixture.endpoint.ownership.requestCapacityAvailable(2, 2));
            verify(fixture.runtime.events(), times(0))
                    .onQueueEvent(
                            any(),
                            org.mockito.Mockito.eq(
                                    org.flexlb.balance.scheduler.AbstractRequestScheduler.QueueKind
                                            .PREEMPTED),
                            any(),
                            org.mockito.Mockito.isNull());
            assertFalse(
                    fixture.endpoint.control(
                            item(config, fixture.endpoint, 1L), WorkerBatcher.Control.REMOVE));
            assertTrue(fixture.endpoint.control(routes.getFirst(), WorkerBatcher.Control.REMOVE));
            assertEquals(7, fixture.endpoint.ownership.captureQueue(20).items().size());
        }
    }

    @Test
    @Timeout(10)
    void concurrentDeliveryCommitsEnforceLimitAndExactReleaseReopensIt() throws Exception {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        try (Fixture fixture = fixture(config);
                var writers = Executors.newFixedThreadPool(8)) {
            var routes = new ArrayList<RequestRoute>();
            for (long id = 1; id <= 64; id++) {
                var route = item(config, fixture.endpoint, id);
                routes.add(route);
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, route));
            }
            var futures = new ArrayList<java.util.concurrent.Future<RequestRoute>>();
            for (var route : routes) {
                futures.add(
                        writers.submit(
                                () -> {
                                    try (var handoff =
                                            fixture.endpoint.commit(
                                                    new PrefillState.CommitPlan(
                                                            PrefillState.CommitKind.QUEUED,
                                                            List.of(route),
                                                            List.of(),
                                                            null,
                                                            new long[] {100L},
                                                            null,
                                                            System.currentTimeMillis()))) {
                                        return handoff == null ? null : route;
                                    }
                                }));
            }
            var committed = new ArrayList<RequestRoute>();
            for (var future : futures) {
                var route = future.get();
                if (route != null) committed.add(route);
            }
            assertEquals(4, committed.size());
            assertEquals(60, fixture.endpoint.ownership.captureQueue(100).items().size());
            assertFalse(fixture.endpoint.ownership.requestCapacityAvailable(4, 1));
            var first = committed.getFirst();
            assertFalse(
                    fixture.endpoint.release(
                            null, item(config, fixture.endpoint, first.requestId())));
            assertFalse(fixture.endpoint.ownership.requestCapacityAvailable(4, 1));
            assertTrue(fixture.endpoint.release(null, first));
            assertFalse(fixture.endpoint.release(null, first));
            assertTrue(fixture.endpoint.ownership.requestCapacityAvailable(4, 1));
            var next = fixture.endpoint.ownership.captureQueue(1).head();
            try (var handoff =
                    fixture.endpoint.commit(
                            new PrefillState.CommitPlan(
                                    PrefillState.CommitKind.QUEUED,
                                    List.of(next),
                                    List.of(),
                                    null,
                                    new long[] {100L},
                                    null,
                                    System.currentTimeMillis()))) {
                assertNotNull(handoff);
            }
            assertFalse(fixture.endpoint.ownership.requestCapacityAvailable(4, 1));
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void staleQueuedCommitReturnsNoHandoffAndDoesNotConsumeRequestCapacity(boolean expired) {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        try (Fixture fixture = fixture(config)) {
            var route = item(config, fixture.endpoint, 1L);
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, route));
            if (expired) {
                ReflectionTestUtils.setField(
                        route.ctx().data(),
                        "schedulingMetadata",
                        SchedulingMetadata.explicit(50, System.currentTimeMillis() - 1L));
            } else {
                assertTrue(fixture.endpoint.control(route, WorkerBatcher.Control.REMOVE));
            }
            assertNull(
                    fixture.endpoint.commit(
                            new PrefillState.CommitPlan(
                                    PrefillState.CommitKind.QUEUED,
                                    List.of(route),
                                    List.of(),
                                    null,
                                    new long[] {100L},
                                    null,
                                    System.currentTimeMillis())));
            assertTrue(fixture.endpoint.ownership.requestCapacityAvailable(1, 1));
            assertEquals(0, fixture.endpoint.ownership.stats().batchCount());
        }
    }

    private static FlexlbConfig preemptionConfig() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getScheduler().setOrdering(ConfigTestSupport.priority());
        return config;
    }

    @Test
    @Timeout(10)
    void eligibilityFailureDoesNotPublishAFreshQueueEntry() {
        FlexlbConfig config = preemptionConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus())
                .thenReturn(
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test"));
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(mock(PrefillTimePredictor.Evaluator.class));
        ReflectionTestUtils.setField(endpoint, "predictor", predictor);
        var requests = EndpointTestSupport.requestRuntime();
        WorkerBatcher runtime =
                WorkerBatcherTestSupport.create(
                        "failed-reservation",
                        endpoint,
                        config,
                        EndpointTestSupport.routeStrategy(requests),
                        requests.events());
        IllegalStateException failure = new IllegalStateException("reservation failed");
        runtime.start();
        try {
            RequestRoute previous = item(config, endpoint, 1L);
            RequestRoute incoming = org.mockito.Mockito.spy(item(config, endpoint, 2L, 90));
            doAnswer(
                            invocation -> {
                                assertEquals(
                                        1L,
                                        WorkerBatcherTestSupport.state(runtime)
                                                .admissionSummary(0, 0L)
                                                .occupiedRequests(),
                                        "eligibility failure must happen before queue publication");
                                throw failure;
                            })
                    .when(incoming)
                    .requestId();
            assertTrue(runtime.offer(previous));
            assertSame(
                    failure,
                    assertThrows(IllegalStateException.class, () -> runtime.offer(incoming)));
            assertEquals(List.of(previous), WorkerBatcherTestSupport.capture(runtime).items());
            assertEquals(
                    1L,
                    WorkerBatcherTestSupport.state(runtime)
                            .admissionSummary(0, 0L)
                            .occupiedRequests());
            verify(requests.events(), times(0))
                    .onQueueEvent(
                            any(),
                            org.mockito.Mockito.eq(
                                    org.flexlb.balance.scheduler.AbstractRequestScheduler.QueueKind
                                            .PREEMPTED),
                            any(),
                            org.mockito.Mockito.isNull());
            RequestRoute next = item(config, endpoint, 3L);
            assertTrue(runtime.offer(next));
            assertEquals(
                    List.of(previous, next), WorkerBatcherTestSupport.capture(runtime).items());
            assertEquals(
                    2L,
                    WorkerBatcherTestSupport.state(runtime)
                            .admissionSummary(0, 0L)
                            .occupiedRequests());
        } finally {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    @Timeout(10)
    void retirementRejectsQueuePublicationBeforeOccupyingCapacity() {
        FlexlbConfig config = preemptionConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus())
                .thenReturn(
                        WorkerStatus.createDiscovered(
                                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test"));
        var requests = EndpointTestSupport.requestRuntime();
        WorkerBatcher runtime =
                WorkerBatcherTestSupport.create(
                        "retiring-publication",
                        endpoint,
                        config,
                        EndpointTestSupport.routeStrategy(requests),
                        requests.events());
        doAnswer(
                        invocation -> {
                            assertEquals(
                                    0L,
                                    WorkerBatcherTestSupport.state(runtime)
                                            .admissionSummary(0, 0L)
                                            .occupiedRequests(),
                                    "retirement rejects queue publication before occupying a request seat");
                            return true;
                        })
                .when(endpoint)
                .isGenerationRetiringOrRetired();
        runtime.start();
        try {
            RequestRoute item = item(config, endpoint, 1L);
            assertFalse(runtime.offer(item));
            assertEquals(
                    0L,
                    WorkerBatcherTestSupport.state(runtime)
                            .admissionSummary(0, 0L)
                            .occupiedRequests());
            assertTrue(WorkerBatcherTestSupport.capture(runtime).items().isEmpty());
            assertFalse(endpoint.release(null, item));
        } finally {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    @Timeout(10)
    void configuredBatchGroupSizeIsIndependentOfConcurrentBatchLimit() throws Exception {
        FlexlbConfig config = productionConfig();
        var decision = config.getScheduler().getDecision();
        decision.setMaxRequests(2);
        decision.setMaxCollectionWaitMs(60_000L);
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        try (Fixture fixture = fixture(config)) {
            CountDownLatch prepared = new CountDownLatch(1);
            AtomicInteger groupSize = new AtomicInteger();
            fixture.endpoint.enableQueueRuntime(
                    org.flexlb.balance.scheduler.QueueExecutionSettings.capture(config));
            var worker = EndpointTestSupport.batcher(fixture.endpoint);
            org.flexlb.balance.scheduler.DeliveryTestSupport.selectWith(
                    worker,
                    (items, reason, evaluator, prediction, real) -> {
                        groupSize.set(items.size());
                        prepared.countDown();
                        return CapacityBoundary.deliveryUnavailable(
                                CapacityBoundary.Resource.PREFILL_BATCH);
                    });
            assertEquals(
                    2,
                    fixture.endpoint
                            .captureRouteProjectionInputs()
                            .queue()
                            .constraints()
                            .maxRequests());
            assertTrue(
                    EndpointTestSupport.offer(
                            fixture.endpoint, item(config, fixture.endpoint, 1L)));
            assertFalse(prepared.await(100L, TimeUnit.MILLISECONDS));
            assertTrue(
                    EndpointTestSupport.offer(
                            fixture.endpoint, item(config, fixture.endpoint, 2L)));
            assertTrue(prepared.await(5L, TimeUnit.SECONDS));
            assertEquals(
                    2, groupSize.get(), "concurrent batch limit must not change each group's size");
        }
    }

    @Test
    void defaultBatchLimitAllowsTwoReservationsAndReopensAfterRelease() {
        FlexlbConfig config = productionConfig();
        try (Fixture fixture = fixture(config)) {
            var items = new ArrayList<RequestRoute>();
            var reservations = new ArrayList<PrefillState.BatchReservation>();
            try {
                for (long id = 1L; id <= 3L; id++) {
                    RequestRoute queued = item(config, fixture.endpoint, id);
                    items.add(queued);
                    assertEquals(2, config.getDispatcher().getMaxInflightPerPrefillWorker());
                    assertTrue(
                            EndpointTestSupport.offer(fixture.endpoint, queued),
                            "BATCH does not impose an additional request-count limit on the waiting queue");
                }
                for (int index = 0; index < 2; index++) {
                    var result =
                            fixture.endpoint.reserveBatch(
                                    items.get(index),
                                    100L + index,
                                    config.getDispatcher().getMaxInflightPerPrefillWorker());
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                    reservations.add(result.reservation());
                }
                assertEquals(
                        PrefillState.CapacityStatus.CAPACITY_FULL,
                        fixture.endpoint.reserveBatch(items.get(2), 102L, 2).status());
                assertEquals(2, fixture.endpoint.ownership.progress().batchSlots());
                var released = fixture.endpoint.ownership.release(reservations.removeFirst(), null);
                assertEquals(PrefillState.RequestRelease.RESERVED, released.kind());
                assertNotNull(released.generationHandoff());
                released.generationHandoff().close();
                assertEquals(1, fixture.endpoint.ownership.progress().batchSlots());
                var replacement = fixture.endpoint.reserveBatch(items.get(2), 102L, 2);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, replacement.status());
                reservations.add(replacement.reservation());
            } finally {
                reservations.forEach(
                        reservation -> assertTrue(fixture.endpoint.release(reservation, null)));
                assertEquals(0, fixture.endpoint.ownership.progress().batchSlots());
            }
        }
    }

    private static FlexlbConfig productionConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        config.getRequestLifecycle().getDecision().setLifetime(2.0);
        return config;
    }

    private static RequestRoute item(
            FlexlbConfig config, PrefillEndpoint endpoint, long requestId) {
        return item(config, endpoint, requestId, 50);
    }

    private static RequestRoute item(
            FlexlbConfig config, PrefillEndpoint endpoint, long requestId, int priority) {
        Request request = Request.builder().requestId(requestId).seqLen(100L).build();
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.data().setFuture(new CompletableFuture<Response>());
        return SchedulingTestConfig.createRoute(
                context, null, null, null, endpoint, null, null, System.currentTimeMillis());
    }

    private static Fixture fixture(FlexlbConfig config) {
        WorkerStatus status =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        EndpointTestSupport.publishStatus(status, response);
        var runtime = EndpointTestSupport.requestRuntime();
        DeliveryDispatcher parked = EndpointTestSupport.routeStrategy(runtime);
        DeliveryDispatcher delivery = org.mockito.Mockito.spy(parked);
        PrefillEndpoint endpoint =
                EndpointTestSupport.prefill(
                        status,
                        config,
                        delivery,
                        org.flexlb.balance.scheduler.SchedulerTestSupport.repository(
                                runtime.events()),
                        mock(DeliveryMetricsReporter.class));
        return new Fixture(endpoint, delivery, parked, runtime);
    }

    private record Fixture(
            PrefillEndpoint endpoint,
            DeliveryDispatcher delivery,
            DeliveryDispatcher parked,
            EndpointTestSupport.TestRequestRuntime runtime)
            implements AutoCloseable {
        public void close() {
            endpoint.close();
            endpoint.awaitRetirement();
        }
    }
}
