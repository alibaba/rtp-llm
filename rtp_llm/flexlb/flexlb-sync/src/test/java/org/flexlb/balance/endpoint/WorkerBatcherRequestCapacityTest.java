package org.flexlb.balance.endpoint;

import static org.flexlb.balance.scheduler.SchedulingTestConfig.freezeInputs;

import org.flexlb.balance.scheduler.AbstractRequestScheduler;
import org.flexlb.balance.scheduler.WorkerBatcherTestSupport;
import org.flexlb.balance.scheduler.DeliveryStrategy;
import org.flexlb.balance.scheduler.DeliveryTransaction;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.springframework.test.util.ReflectionTestUtils;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.AdditionalAnswers.delegatesTo;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** The configured limit, real worker publication lock and canonical request ledger together. */
class WorkerBatcherRequestCapacityTest {

    @ParameterizedTest
    @CsvSource({"false,90,false", "true,50,false", "true,0,false", "true,90,true"})
    void queuedPreemptionHonorsDisabledPolicyAndStrictlyHigherPriority(
            boolean enabled, int priority, boolean accepted) {
        FlexlbConfig config = preemptionConfig();
        if (!enabled) { config.priorityOrdering().setPreemption(null); }
        try (Fixture fixture = fixture(config)) {
            RequestRoute first = item(config, fixture.endpoint, 1L);
            RequestRoute second = item(config, fixture.endpoint, 2L);
            RequestRoute incoming = item(config, fixture.endpoint, 3L, priority);
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, first));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, second));
            assertEquals(accepted, EndpointTestSupport.offer(fixture.endpoint, incoming));
            assertEquals(2L, fixture.endpoint.admissionSummary(0).occupiedRequests());
            assertEquals(accepted ? List.of(incoming, first) : List.of(first, second),
                    WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.endpoint)).items());
            verify(fixture.runtime.events(), times(accepted ? 1 : 0)).onQueuedItemPreempted(second, incoming);
            assertFalse(fixture.endpoint.removeQueued(accepted ? second : incoming, "non-owner cleanup"));
            assertEquals(2L, fixture.endpoint.admissionSummary(0).occupiedRequests());
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void failedReplacementLeavesVictimsUntouchedForRetry(boolean fatal) {
        FlexlbConfig config = preemptionConfig();
        try (Fixture fixture = fixture(config)) {
            RequestRoute first = item(config, fixture.endpoint, 1L);
            RequestRoute second = item(config, fixture.endpoint, 2L);
            RequestRoute incoming = org.mockito.Mockito.spy(item(config, fixture.endpoint, 3L, 90));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, first));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, second));
            doAnswer(invocation -> {
                if (fatal) { throw new AssertionError("injected admission failure"); }
                throw new IllegalStateException("injected admission failure");
            }).when(incoming).requiresRouteReservation();
            Class<? extends Throwable> expected = fatal ? AssertionError.class : IllegalStateException.class;
            assertThrows(expected,
                    () -> EndpointTestSupport.offer(fixture.endpoint, incoming));
            assertEquals(2L, fixture.endpoint.admissionSummary(0).occupiedRequests());
            assertEquals(List.of(first, second),
                    WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.endpoint)).items());
            verify(fixture.runtime.events(), times(0)).onQueuedItemPreempted(any(), any());

            org.mockito.Mockito.doCallRealMethod().when(incoming).requiresRouteReservation();
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, incoming));
            assertEquals(2L, fixture.endpoint.admissionSummary(0).occupiedRequests());
            assertEquals(List.of(incoming, first),
                    WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.endpoint)).items());
            verify(fixture.runtime.events()).onQueuedItemPreempted(second, incoming);
            assertFalse(fixture.endpoint.removeQueued(second, "late victim cleanup"));
            assertTrue(fixture.endpoint.removeQueued(incoming, "replacement cleanup"));
            assertTrue(fixture.endpoint.removeQueued(first, "remaining cleanup"));
            assertEquals(0L, fixture.endpoint.admissionSummary(0).occupiedRequests());
        }
    }

    @Test
    void insufficientEligibleVictimsLeaveAllQueuedOwnersUntouched() {
        FlexlbConfig config = preemptionConfig();
        try (Fixture fixture = fixture(config)) {
            RequestRoute first = org.mockito.Mockito.spy(item(config, fixture.endpoint, 1L));
            RequestRoute second = item(config, fixture.endpoint, 2L);
            RequestRoute incoming = item(config, fixture.endpoint, 3L, 90);
            PrefillState state = (PrefillState) org.springframework.test.util.ReflectionTestUtils.getField(fixture.endpoint, "prefillState");
            // Hold the real ownership lock across setup so the worker cannot consume the test victims.
            state.ownershipLock().lock();
            try {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, first));
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, second));
                // Two seats are required; an equal-priority request cannot be a victim.
                org.mockito.Mockito.doReturn(90).when(first).priority();
                assertTrue(fixture.endpoint.replaceQueuedRoutesLocked(incoming, 1L).isEmpty());
                assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
                assertEquals(List.of(first, second), state.captureQueue(Integer.MAX_VALUE).items());
                org.mockito.Mockito.doCallRealMethod().when(first).priority();
                assertEquals(List.of(second, first), fixture.endpoint.replaceQueuedRoutesLocked(incoming, 1L));
                assertEquals(List.of(incoming), state.captureQueue(Integer.MAX_VALUE).items());
                assertEquals(1L, state.admissionSummary(0, 0L).occupiedRequests());
                assertFalse(state.removeQueuedLocked(first));
                assertFalse(state.removeQueuedLocked(second));
                assertTrue(state.removeQueuedLocked(incoming));
                assertEquals(0L, state.admissionSummary(0, 0L).occupiedRequests());
            } finally {
                state.ownershipLock().unlock();
            }
        }
    }

    @Test
    void completedFrontendFutureDoesNotPreventReclaimingAnExactWaitingSeat() {
        FlexlbConfig config = preemptionConfig();
        try (Fixture fixture = fixture(config)) {
            RequestRoute completed = item(config, fixture.endpoint, 1L, 10);
            RequestRoute waiting = item(config, fixture.endpoint, 2L, 50);
            RequestRoute incoming = item(config, fixture.endpoint, 3L, 90);
            PrefillState state = (PrefillState) ReflectionTestUtils.getField(fixture.endpoint, "prefillState");
            state.ownershipLock().lock();
            try {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, completed));
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, waiting));
                Response frontendResult = Response.error(org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED);
                assertTrue(completed.future().complete(frontendResult));
                assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests(), "frontend publication does not remove the waiting resource seat");
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, incoming));
                assertEquals(List.of(incoming, waiting), state.captureQueue(Integer.MAX_VALUE).items());
                assertEquals(2L, state.admissionSummary(0, 0L).occupiedRequests());
                assertSame(frontendResult, completed.future().join(), "reclaiming queue ownership cannot rewrite an already published result");
                assertFalse(state.removeQueuedLocked(completed));
                verify(fixture.runtime.events()).onQueuedItemPreempted(completed, incoming);
            } finally {
                state.ownershipLock().unlock();
            }
        }
    }

    @Test
    @Timeout(10)
    void eligibilityFailureDoesNotPublishAFreshQueueEntry() {
        FlexlbConfig config = preemptionConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(WorkerStatus.createDiscovered(
                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test"));
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.evaluator()).thenReturn(mock(PrefillTimePredictor.Evaluator.class));
        when(endpoint.getPredictor()).thenReturn(predictor);
        var requests = EndpointTestSupport.requestRuntime();
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("failed-reservation", endpoint, config,
                EndpointTestSupport.routeStrategy(requests), requests.events());
        IllegalStateException failure = new IllegalStateException("reservation failed");
        runtime.start();
        try {
            RequestRoute previous = item(config, endpoint, 1L);
            RequestRoute incoming = org.mockito.Mockito.spy(item(config, endpoint, 2L, 90));
            doAnswer(invocation -> {
                assertEquals(1L, WorkerBatcherTestSupport.state(runtime).admissionSummary(0, 0L).occupiedRequests(),
                        "eligibility failure must happen before queue publication");
                throw failure;
            }).when(incoming).requiresRouteReservation();
            assertTrue(runtime.offer(previous));
            assertSame(failure, assertThrows(IllegalStateException.class, () -> runtime.offer(incoming)));
            assertEquals(List.of(previous), WorkerBatcherTestSupport.capture(runtime).items());
            assertEquals(1L, WorkerBatcherTestSupport.state(runtime).admissionSummary(0, 0L).occupiedRequests());
            verify(requests.events(), times(0)).onQueuedItemPreempted(any(), any());
            RequestRoute next = item(config, endpoint, 3L);
            assertTrue(runtime.offer(next));
            assertEquals(List.of(previous, next), WorkerBatcherTestSupport.capture(runtime).items());
            assertEquals(2L, WorkerBatcherTestSupport.state(runtime).admissionSummary(0, 0L).occupiedRequests());
        } finally {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    @Timeout(10)
    void retirementRejectsQueuePublicationBeforeOccupyingCapacity() {
        FlexlbConfig config = preemptionConfig();
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getStatus()).thenReturn(WorkerStatus.createDiscovered(
                RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test"));
        var requests = EndpointTestSupport.requestRuntime();
        WorkerBatcher runtime = WorkerBatcherTestSupport.create("retiring-publication", endpoint, config,
                EndpointTestSupport.routeStrategy(requests), requests.events());
        doAnswer(invocation -> {
            assertEquals(0L, WorkerBatcherTestSupport.state(runtime).admissionSummary(0, 0L).occupiedRequests(),
                    "retirement rejects queue publication before occupying a request seat");
            return true;
        }).when(endpoint).isGenerationRetiringOrRetired();
        runtime.start();
        try {
            RequestRoute item = item(config, endpoint, 1L);
            assertFalse(runtime.offer(item));
            assertEquals(0L, WorkerBatcherTestSupport.state(runtime).admissionSummary(0, 0L).occupiedRequests());
            assertTrue(WorkerBatcherTestSupport.capture(runtime).items().isEmpty());
            assertFalse(endpoint.releaseRequest(item));
        } finally {
            assertNull(runtime.stopAndAwait());
        }
    }

    @Test
    @Timeout(10)
    void concurrentHigherPriorityOffersTransferOnlyTheTwoQueuedSeats() throws Exception {
        FlexlbConfig config = preemptionConfig();
        try (Fixture fixture = fixture(config); var writers = Executors.newFixedThreadPool(8)) {
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 1L)));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 2L)));
            var results = new ArrayList<java.util.concurrent.Future<Boolean>>();
            for (long id = 3L; id < 35L; id++) {
                RequestRoute incoming = item(config, fixture.endpoint, id, 90);
                results.add(writers.submit(() -> EndpointTestSupport.offer(fixture.endpoint, incoming)));
            }
            int accepted = 0;
            for (var result : results) { if (result.get()) { accepted++; } }
            assertEquals(2, accepted);
            assertEquals(2L, fixture.endpoint.admissionSummary(0).occupiedRequests());
            assertTrue(WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.endpoint)).items().stream().allMatch(item -> item.priority() == 90));
            verify(fixture.runtime.events(), times(2)).onQueuedItemPreempted(any(), any());
        }
    }

    @Test
    void committedPrefillWorkCannotBeReclaimedAsQueuedCapacity() {
        FlexlbConfig config = preemptionConfig();
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        try (Fixture fixture = fixture(config)) {
            RequestRoute committed = item(config, fixture.endpoint, 1L);
            {
                var reservation = EndpointTestSupport.reserveUnqueued(fixture.endpoint, committed, 100L);
                try (var preparationReservation = EndpointTestSupport.preparation(reservation);
                     var commit = fixture.endpoint.tryBeginRouteCommitAdmission();
                     var handoff = commit.commit(List.of(committed), List.of(reservation))) {
                    assertFalse(fixture.endpoint.canPreemptQueuedRequest(90));
                    assertFalse(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 2L, 90)));
                    assertEquals(1L, fixture.endpoint.admissionSummary(0).occupiedRequests());
                    assertTrue(WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.endpoint)).items().isEmpty());
                    verify(fixture.runtime.events(), times(0)).onQueuedItemPreempted(any(), any());
                }
            }
        }
    }

    @Test
    @Timeout(10)
    void preemptingPreparedRequestInvalidatesItsDeliveryAndPreservesReplacement() throws Exception {
        FlexlbConfig config = preemptionConfig();
        config.getScheduler().setDecision(DecisionPolicyConfig.single());
        config.getDispatcher().setMaxInflightPerPrefillWorker(1);
        CountDownLatch prepared = new CountDownLatch(1);
        CountDownLatch resume = new CountDownLatch(1);
        CountDownLatch replacementSelected = new CountDownLatch(1);
        try (Fixture fixture = fixture(config)) {
            DeliveryStrategy live = EndpointTestSupport.liveRouteStrategy(fixture.runtime);
            doAnswer(invocation -> {
                List<RequestRoute> candidates = invocation.getArgument(0);
                if (candidates.getFirst().requestId() == 1L) {
                    DeliveryTransaction transaction = live.prepare(candidates,
                            invocation.getArgument(1), invocation.getArgument(2));
                    assertEquals(1, transaction.items().size());
                    prepared.countDown();
                    assertTrue(resume.await(5, TimeUnit.SECONDS));
                    return transaction;
                }
                replacementSelected.countDown();
                return fixture.parked.prepare(candidates, invocation.getArgument(1), invocation.getArgument(2));
            }).when(fixture.delivery).prepare(anyList(), any(), any());
            RequestRoute victim = item(config, fixture.endpoint, 1L);
            RequestRoute incoming = item(config, fixture.endpoint, 2L, 90);
            try {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, victim));
                assertTrue(prepared.await(5, TimeUnit.SECONDS));
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, incoming));
            } finally {
                resume.countDown();
            }
            assertTrue(replacementSelected.await(5, TimeUnit.SECONDS));
            assertEquals(List.of(incoming), WorkerBatcherTestSupport.capture(EndpointTestSupport.batcher(fixture.endpoint)).items());
            assertEquals(1L, fixture.endpoint.admissionSummary(0).occupiedRequests());
            assertTrue(org.mockito.Mockito.mockingDetails(fixture.runtime.requests()).getInvocations().stream()
                    .noneMatch(call -> call.getMethod().getName().equals("claimRouteDelivery")));
            verify(fixture.runtime.events()).onQueuedItemPreempted(victim, incoming);
        }
    }

    private static FlexlbConfig preemptionConfig() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getScheduler().setOrdering(QueueOrderingConfig.priority());
        return config;
    }

    @Test
    @Timeout(10)
    void nonBatchConcurrentOffersShareOneCountLimitAndExactCancellationReopensIt() throws Exception {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getDispatcher().setMaxInflightPerPrefillWorker(4);
        config.getScheduler().getDecision().setMaxRequests(1);
        try (Fixture fixture = fixture(config);
             var writers = Executors.newFixedThreadPool(8)) {
            var results = new ArrayList<java.util.concurrent.Future<RequestRoute>>();
            for (long id = 1L; id <= 64L; id++) {
                RequestRoute item = item(config, fixture.endpoint, id);
                results.add(writers.submit(() -> EndpointTestSupport.offer(fixture.endpoint, item) ? item : null));
            }
            var admitted = new ArrayList<RequestRoute>();
            for (var result : results) {
                RequestRoute owned = result.get();
                if (owned != null) { admitted.add(owned); }
            }
            assertEquals(4, admitted.size(), "each writer attempts once; publication must not oversubscribe");
            assertEquals(4L, fixture.endpoint.admissionSummary(0).occupiedRequests());
            RequestRoute first = admitted.getFirst();
            assertFalse(fixture.endpoint.removeQueued(item(config, fixture.endpoint, first.requestId()), "stale identity"));
            assertFalse(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 70L)));
            assertTrue(fixture.endpoint.removeQueued(first, "cancel exact queued request"));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 71L)));
            assertEquals(4L, fixture.endpoint.admissionSummary(0).occupiedRequests());
        }
    }

    @Test
    void singleDecisionUsesTheDefaultTwoRequestLimit() {
        FlexlbConfig config = productionConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getScheduler().setDecision(DecisionPolicyConfig.single());
        try (Fixture fixture = fixture(config)) {
            for (long requestId = 1L; requestId <= 2L; requestId++) {
                assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, requestId)));
            }
            assertEquals(2L, fixture.endpoint.admissionSummary(0).occupiedRequests());
            assertFalse(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 3L)));
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
            doAnswer(invocation -> {
                List<RequestRoute> items = invocation.getArgument(0);
                groupSize.set(items.size());
                prepared.countDown();
                return fixture.parked.prepare(items, invocation.getArgument(1), invocation.getArgument(2));
            }).when(fixture.delivery).prepare(anyList(), any(), any());
            fixture.endpoint.enableQueueRuntime(org.flexlb.balance.scheduler.QueueExecutionSettings.capture(config));
            assertEquals(2, fixture.endpoint.captureRouteProjectionInputs().queue().constraints().maxRequests());
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 1L)));
            assertFalse(prepared.await(100L, TimeUnit.MILLISECONDS));
            assertTrue(EndpointTestSupport.offer(fixture.endpoint, item(config, fixture.endpoint, 2L)));
            assertTrue(prepared.await(5L, TimeUnit.SECONDS));
            assertEquals(2, groupSize.get(), "concurrent batch limit must not change each group's size");
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
                    assertEquals(2, queued.requirements().maxInflightBatchesPerPrefillWorker());
                    assertTrue(EndpointTestSupport.offer(fixture.endpoint, queued),
                            "BATCH does not impose an additional request-count limit on the waiting queue");
                }
                for (int index = 0; index < 2; index++) {
                    var result = fixture.endpoint.reserveBatch(items.get(index), 100L + index,
                            items.get(index).requirements().maxInflightBatchesPerPrefillWorker());
                    assertEquals(PrefillState.CapacityStatus.ACQUIRED, result.status());
                    reservations.add(result.reservation());
                }
                assertEquals(PrefillState.CapacityStatus.CAPACITY_FULL,
                        fixture.endpoint.reserveBatch(items.get(2), 102L, 2).status());
                fixture.endpoint.rollbackReservation(reservations.removeFirst());
                var replacement = fixture.endpoint.reserveBatch(items.get(2), 102L, 2);
                assertEquals(PrefillState.CapacityStatus.ACQUIRED, replacement.status());
                reservations.add(replacement.reservation());
            } finally {
                reservations.forEach(fixture.endpoint::rollbackReservation);
            }
        }
    }

    private static FlexlbConfig productionConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        config.getRequestLifecycle().getDecision().setLifetime(2.0);
        return config;
    }

    private static RequestRoute item(FlexlbConfig config, PrefillEndpoint endpoint, long requestId) {
        return item(config, endpoint, requestId, 50);
    }

    private static RequestRoute item(FlexlbConfig config, PrefillEndpoint endpoint, long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(100L);
        RequestContext context = new RequestContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(priority, Long.MAX_VALUE));
        context.setFuture(new CompletableFuture<Response>());
        return org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(freezeInputs(context), null, null, null,
                endpoint, null, null, System.currentTimeMillis());
    }

    private static Fixture fixture(FlexlbConfig config) {
        WorkerStatus status = WorkerStatus.createDiscovered(RoleType.PREFILL, "test", "127.0.0.1", 8080, 9090, "test");
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        EndpointTestSupport.publishStatus(status, response);
        var runtime = EndpointTestSupport.requestRuntime();
        DeliveryStrategy parked = EndpointTestSupport.routeStrategy(runtime);
        DeliveryStrategy delivery = mock(DeliveryStrategy.class, delegatesTo(parked));
        PrefillEndpoint endpoint = EndpointTestSupport.prefill(status, config, delivery, org.flexlb.balance.scheduler.SchedulerTestSupport.repository(runtime.events()), mock(DeliveryMetricsReporter.class));
        return new Fixture(endpoint, delivery, parked, runtime);
    }

    private record Fixture(PrefillEndpoint endpoint, DeliveryStrategy delivery, DeliveryStrategy parked,
                           EndpointTestSupport.TestRequestRuntime runtime)
            implements AutoCloseable {
        public void close() { endpoint.close(); endpoint.awaitRetirement(); }
    }
}
