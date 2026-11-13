package org.flexlb.balance.schedule;

import org.flexlb.balance.metrics.DeliveryMetricsReporter;
import org.flexlb.balance.queue.SchedulingQueue;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.CapacityNotifierTestSupport;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;
import org.flexlb.worker.status.WorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongConsumer;

import static org.flexlb.balance.schedule.RequestProtocolTestSupport.await;
import static org.flexlb.balance.schedule.RequestProtocolTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies global queue progress and capacity wake-ups under concurrent placement.
 */
class GlobalQueueProgressTest {

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preemptionSlotReleaseCannotBeLostBeforeCapacityParking(boolean releaseBeforeParking)
            throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            var pending = new ConcurrentHashMap<Long, CompletableFuture<RequestResult>>();
            var attempts = new ConcurrentHashMap<Long, AtomicInteger>();
            var beforeParking = new CountDownLatch(1);
            var allowParking = new CountDownLatch(1);
            var decode = SchedulerTestSupport.mockDecodeEndpoint();
            f.onAdmission =
                    id -> {
                        var route = f.routes.get(id);
                        when(route.decodeEndpoint()).thenReturn(decode);
                        when(decode.registrationState(any()))
                                .thenReturn(DecodeResources.RegistrationState.UNCONFIRMED);
                        when(route.requestId()).thenReturn(String.valueOf(id));
                        if (attempts.computeIfAbsent(id, ignored -> new AtomicInteger())
                                        .incrementAndGet()
                                == 1) {
                            f.admissionBlocked.add(id);
                            if (id == 1003L && releaseBeforeParking) {
                                var finishingScheduler =
                                        RequestProtocolTestSupport.placementScheduler(
                                                f.mutations.get(id));
                                doAnswer(
                                                call -> {
                                                    beforeParking.countDown();
                                                    await(allowParking);
                                                    return null;
                                                })
                                        .when(finishingScheduler)
                                        .finishSelectionWithoutCommit(
                                                org.mockito.ArgumentMatchers.same(
                                                        f.mutations.get(id)),
                                                any());
                            }
                        } else {
                            f.admissionBlocked.remove(id);
                        }
                    };
            when(f.eviction.tryPreempt(any(), eq(decode)))
                    .thenAnswer(
                            call -> {
                                RequestContext context = call.getArgument(0);
                                var result = new CompletableFuture<RequestResult>();
                                pending.put(Long.parseLong(context.getRequestId()), result);
                                return result;
                            });
            try {
                f.submit(1001L, "a");
                awaitCondition(() -> pending.containsKey(1001L));
                f.submit(1002L, "a");
                awaitCondition(() -> pending.containsKey(1002L));
                f.submit(1003L, "a");
                if (releaseBeforeParking) {
                    await(beforeParking);
                } else {
                    awaitCondition(() -> f.scheduler.blockedRequestCount() > 0);
                    awaitCondition(
                            () -> {
                                var queueLock =
                                        (ReentrantLock)
                                                ReflectionTestUtils.getField(f.scheduler, "lock");
                                queueLock.lock();
                                try {
                                    return (((SchedulingRequests)
                                                            ReflectionTestUtils.getField(
                                                                    f.scheduler,
                                                                    "schedulingRequests"))
                                                    .decisionCount()
                                            == 0);
                                } finally {
                                    queueLock.unlock();
                                }
                            });
                    assertEquals(
                            1,
                            f.scheduler.blockedRequestCount(),
                            "one request can wait for either capacity or a preemption slot");
                }
                pending.get(1001L).complete(new RequestResult(null, false, "no victim"));
                awaitCondition(
                        () -> {
                            var queueLock =
                                    (ReentrantLock)
                                            ReflectionTestUtils.getField(f.scheduler, "lock");
                            queueLock.lock();
                            try {
                                return ((SchedulingRequests)
                                                        ReflectionTestUtils.getField(
                                                                f.scheduler, "schedulingRequests"))
                                                .preemptionCount()
                                        == 1;
                            } finally {
                                queueLock.unlock();
                            }
                        });
                allowParking.countDown();
                awaitCondition(() -> f.admitted.contains(1003L));
                assertEquals(2, attempts.get(1003L).get());
            } finally {
                allowParking.countDown();
                pending.values()
                        .forEach(
                                result ->
                                        result.complete(
                                                new RequestResult(null, false, "test finished")));
            }
        }
    }

    @Test
    void releasingBlockedPlanPublishesCapacityBeforeParking() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            AtomicInteger attempts = new AtomicInteger();
            AtomicInteger closes = new AtomicInteger();
            f.onSelection =
                    id -> {
                        if (attempts.get() > 0) {
                            assertEquals(
                                    1,
                                    closes.get(),
                                    "retry cannot begin before the old route closes");
                        }
                    };
            f.onAdmission =
                    id -> {
                        if (attempts.getAndIncrement() == 0) {
                            f.admissionBlocked.add(id);
                            RequestResult route = f.routes.get(id);
                            doAnswer(
                                            invocation -> {
                                                closes.incrementAndGet();
                                                f.admissionBlocked.remove(id);
                                                f.release("a");
                                                return null;
                                            })
                                    .when(
                                            RequestProtocolTestSupport.placementScheduler(
                                                    f.mutations.get(id)))
                                    .finishSelectionWithoutCommit(
                                            org.mockito.ArgumentMatchers.same(f.mutations.get(id)),
                                            any());
                        }
                    };
            f.submit(993L, "b");
            awaitCondition(() -> f.admitted.contains(993L));
            assertEquals(
                    2, attempts.get(), "the cleanup edge must grant a retry without another event");
            assertEquals(1, closes.get());
        }
    }

    @Test
    void parkedPlanRetainsSelectionDiagnosticsAfterClosing() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            Map<String, Object> diagnostics = Map.of("reason", "no eligible worker");
            doReturn(
                            RequestResult.blocked(
                                    new CapacityKey(RoleType.PREFILL, "a", null),
                                    null,
                                    diagnostics))
                    .when(f.router)
                    .select(any(), nullable(String.class));
            f.submit(994L, "a");
            awaitCondition(
                    () -> {
                        var saved = f.contexts.get(994L).getSchedulingDiagnostics();
                        return saved != null && diagnostics.equals(saved.get("decision"));
                    });
            verify(RequestProtocolTestSupport.placementScheduler(f.mutations.get(994L)))
                    .finishSelectionWithoutCommit(
                            org.mockito.ArgumentMatchers.same(f.mutations.get(994L)),
                            org.mockito.ArgumentMatchers.isNull());
        }
    }

    @Test
    void unpublishedSelectionExitFailureReportsFailureAndReleasesPlanningSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.admissionBlocked.add(995L);
            f.onSelection =
                    id -> {
                        if (id == 995L)
                            doThrow(new IllegalStateException("handle close failed"))
                                    .when(
                                            RequestProtocolTestSupport.placementScheduler(
                                                    f.mutations.get(id)))
                                    .finishSelectionWithoutCommit(
                                            org.mockito.ArgumentMatchers.same(f.mutations.get(id)),
                                            org.mockito.ArgumentMatchers.isNull());
                    };
            f.submit(995L, "b");
            awaitCondition(
                    () ->
                            ReflectionTestUtils.getField(f.lifecycle, "requestProcessingFailure")
                                    != null);
            verify(f.lifecycle)
                    .recordRequestFailure(
                            org.mockito.ArgumentMatchers.argThat(
                                    failure -> "handle close failed".equals(failure.getMessage())));
            verify(RequestProtocolTestSupport.placementScheduler(f.mutations.get(995L)))
                    .finishSelectionWithoutCommit(
                            org.mockito.ArgumentMatchers.same(f.mutations.get(995L)),
                            org.mockito.ArgumentMatchers.isNull());
            f.submit(996L, "b");
            awaitCondition(() -> f.admitted.contains(996L));
        }
    }

    @Test
    void routeRollbackFailureIsReportedWithoutStrandingThePlanningSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            var failure = new IllegalStateException("route rollback failed");
            f.onAdmission =
                    id -> {
                        if (id == 997L) {
                            f.admissionBlocked.add(id);
                            var route = f.routes.get(id);
                            var decode = mock(DecodeEndpoint.class);
                            var reservation =
                                    new DecodeEndpoint.RequestRegistration(1L, String.valueOf(id));
                            when(route.decodeEndpoint()).thenReturn(decode);
                            when(route.decodeRegistration()).thenReturn(reservation);
                            when(decode.registrationState(reservation))
                                    .thenReturn(
                                            org.flexlb.worker.endpoint.DecodeResources
                                                    .RegistrationState.UNCONFIRMED);
                            doThrow(failure).when(decode).rollbackSelection(reservation);
                        }
                    };
            f.submit(997L, "b");
            RequestProtocolTestSupport.awaitCondition(
                    () -> SchedulerTestSupport.failure(f.lifecycle) == failure);
            verify(
                            RequestProtocolTestSupport.placementScheduler(f.mutations.get(997L)),
                            timeout(1_000))
                    .finishSelectionWithoutCommit(
                            org.mockito.ArgumentMatchers.same(f.mutations.get(997L)), any());
            f.submit(998L, "b");
            awaitCondition(() -> f.admitted.contains(998L));
        }
    }

    @Test
    void shutdownBeforePlannerSubmissionStillDrainsTheQueuedRequest() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            var original =
                    (java.util.concurrent.ExecutorService)
                            ReflectionTestUtils.getField(f.scheduler, "planners");
            var stopping = mock(java.util.concurrent.ExecutorService.class);
            when(stopping.isShutdown()).thenReturn(true);
            when(stopping.isTerminated()).thenReturn(true);
            var rejected = new CountDownLatch(1);
            doAnswer(
                            call -> {
                                f.scheduler.stop();
                                rejected.countDown();
                                throw new java.util.concurrent.RejectedExecutionException(
                                        "scheduler stopped");
                            })
                    .when(stopping)
                    .execute(any(Runnable.class));
            ReflectionTestUtils.setField(f.scheduler, "planners", stopping);
            original.shutdown();

            f.submit(996L, "b");
            await(rejected);
            assertTrue(f.scheduler.awaitStopped(System.nanoTime() + TimeUnit.SECONDS.toNanos(5)));
            assertEquals(1, RequestProtocolTestSupport.queuedCount(f.scheduler));
            org.junit.jupiter.api.Assertions.assertNull(SchedulerTestSupport.failure(f.lifecycle));
            RequestProtocolTestSupport.close(f.scheduler);
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.scheduler));
            verify(f.router, never()).select(eq(f.contexts.get(996L)), nullable(String.class));
        }
    }

    @Test
    void rejectedPlannerReturnsItsSlotEvenWhenQueueRemovalFails() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            var original =
                    (java.util.concurrent.ExecutorService)
                            ReflectionTestUtils.getField(f.scheduler, "planners");
            var rejecting = mock(java.util.concurrent.ExecutorService.class);
            when(rejecting.isTerminated()).thenReturn(true);
            doThrow(new java.util.concurrent.RejectedExecutionException("rejected"))
                    .when(rejecting)
                    .execute(any(Runnable.class));
            var queue =
                    spy(
                            (SchedulingQueue)
                                    ReflectionTestUtils.getField(f.scheduler, "orderedQueue"));
            doThrow(new IllegalStateException("queue removal failed"))
                    .doCallRealMethod()
                    .when(queue)
                    .remove(any());
            ReflectionTestUtils.setField(f.scheduler, "orderedQueue", queue);
            ReflectionTestUtils.setField(f.scheduler, "planners", rejecting);
            original.shutdown();

            f.submit(996L, "b");
            awaitCondition(() -> SchedulerTestSupport.failure(f.scheduler) != null);
            assertTrue(f.scheduler.awaitStopped(System.nanoTime() + TimeUnit.SECONDS.toNanos(5)));
            assertEquals(
                    0,
                    ((SchedulingRequests)
                                    ReflectionTestUtils.getField(f.scheduler, "schedulingRequests"))
                            .decisionCount());
            assertEquals(0, RequestProtocolTestSupport.queuedCount(f.scheduler));
            verify(f.router, never()).select(any(), nullable(String.class));
        }
    }

    @Test
    void missingPlacementResultDoesNotStrandTheAdmissionOwner() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            doReturn(null).when(f.router).select(any(), nullable(String.class));
            f.submit(999L, "a");
            verify(f.lifecycle, timeout(1_000)).publish(eq(f.contexts.get(999L)), any(), eq(false));
            verify(RequestProtocolTestSupport.placementScheduler(f.mutations.get(999L)))
                    .finishSelectionWithoutCommit(
                            org.mockito.ArgumentMatchers.same(f.mutations.get(999L)),
                            org.mockito.ArgumentMatchers.isNull());
        }
    }

    @Test
    void failedPlanPreparationClosesSelectedRouteAndPreservesItsCause() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.aSlots.set(1);
            var primary = new IllegalStateException("diagnostics failed");
            var cleanup = new IllegalStateException("handle close failed");
            var finishingThread = new java.util.concurrent.atomic.AtomicReference<Thread>();
            f.onSelection =
                    id -> {
                        SelectionAttempt handle = f.mutations.get(id);
                        doThrow(primary)
                                .when(handle.requestContext())
                                .recordRoutingDiagnostics(
                                        org.mockito.ArgumentMatchers.same(handle),
                                        nullable(Map.class));
                        doAnswer(
                                        ignored -> {
                                            finishingThread.set(Thread.currentThread());
                                            throw cleanup;
                                        })
                                .when(RequestProtocolTestSupport.placementScheduler(handle))
                                .finishSelectionWithoutCommit(
                                        org.mockito.ArgumentMatchers.same(handle),
                                        org.mockito.ArgumentMatchers.isNull());
                    };
            f.submit(992L, "a");
            var response = ArgumentCaptor.forClass(Response.class);
            verify(f.lifecycle, timeout(5_000))
                    .publish(eq(f.contexts.get(992L)), response.capture(), eq(false));
            assertEquals(
                    "Placement failed: IllegalStateException: diagnostics failed",
                    response.getValue().getErrorMessage());
            assertEquals(List.of(cleanup), List.of(primary.getSuppressed()));
            assertTrue(
                    finishingThread.get().getName().startsWith("flexlb-global-planner-"),
                    "the selecting task must finish admission cleanup");

            verify(RequestProtocolTestSupport.placementScheduler(f.mutations.get(992L)))
                    .finishSelectionWithoutCommit(
                            org.mockito.ArgumentMatchers.same(f.mutations.get(992L)),
                            org.mockito.ArgumentMatchers.isNull());
            assertFalse(f.admitted.contains(992L));
        }
    }

    @Test
    void schedulerInvalidationOnlyRemovesTheExactGlobalQueueEntry() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            doAnswer(
                            call -> {
                                RequestContext context = call.getArgument(0);
                                CompletableFuture<Response> future =
                                        (CompletableFuture<Response>) call.callRealMethod();
                                f.contexts.put(Long.parseLong(context.getRequestId()), context);
                                f.requests.put(Long.parseLong(context.getRequestId()), future);
                                return future;
                            })
                    .when(f.lifecycle)
                    .submit(any(), any(Runnable.class));
            f.submit(991L, "a");
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(f.scheduler, 1);
            QueuedRequestScheduler queue = (QueuedRequestScheduler) f.scheduler;
            RequestContext context = f.contexts.get(991L);
            RequestProtocolTestSupport.recordCancellation(
                    f.lifecycle,
                    context,
                    CancelReason.CLIENT_CANCELLED,
                    "cancelled before scheduler notification");
            queue.removeRequest(context);
            awaitCondition(() -> RequestProtocolTestSupport.queuedCount(queue) == 0);

            f.aSlots.set(1);
            f.release("a");
            f.submit(992L, "b");
            awaitCondition(() -> f.admitted.contains(992L));
            assertFalse(f.admitted.contains(991L));
        }
    }

    @Test
    void concurrentPublicationDoesNotDelayRefillingAReleasedSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch firstPlanning = new CountDownLatch(1);
            CountDownLatch finishFirstPlanning = new CountDownLatch(1);
            CountDownLatch firstCommitting = new CountDownLatch(1);
            CountDownLatch secondCommitting = new CountDownLatch(1);
            CountDownLatch finishFirst = new CountDownLatch(1);
            CountDownLatch finishSecond = new CountDownLatch(1);
            CountDownLatch thirdPlanning = new CountDownLatch(1);
            f.onAdmission =
                    id -> {
                        if (id == 1L) {
                            firstCommitting.countDown();
                            await(finishFirst);
                        } else if (id == 2L) {
                            secondCommitting.countDown();
                            await(finishSecond);
                        }
                    };
            f.onSelection =
                    id -> {
                        if (id == 1L) {
                            firstPlanning.countDown();
                            await(finishFirstPlanning);
                        } else if (id == 2L) {
                            await(firstCommitting);
                        } else if (id == 3L) {
                            thirdPlanning.countDown();
                        }
                    };
            try {
                f.submit(1, "b");
                await(firstPlanning);
                f.submit(2, "c");
                awaitCondition(() -> f.selected.contains(2L));
                finishFirstPlanning.countDown();
                await(firstCommitting);
                await(secondCommitting);
                f.submit(3, "d");
                finishFirst.countDown();
                await(secondCommitting);
                assertTrue(
                        thirdPlanning.await(1, TimeUnit.SECONDS),
                        "R3 must reuse R1's slot while R2 is still publishing");
            } finally {
                finishFirstPlanning.countDown();
                finishFirst.countDown();
                finishSecond.countDown();
            }
        }
    }

    @Test
    void selectionAndPublicationStayInOneTask() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            AtomicReference<Thread> selectedBy = new AtomicReference<>();
            AtomicReference<Thread> publishedBy = new AtomicReference<>();
            f.onSelection = id -> selectedBy.set(Thread.currentThread());
            f.onAdmission = id -> publishedBy.set(Thread.currentThread());
            f.submit(812L, "b");
            awaitCondition(() -> f.admitted.contains(812L));
            assertEquals(
                    selectedBy.get(),
                    publishedBy.get(),
                    "selected resources must be published before the selecting task returns");
        }
    }

    @Test
    void slowPlannerDoesNotBlockOtherResultsOrRefillingItsPeersSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch finish = new CountDownLatch(1);
            f.onSelection =
                    id -> {
                        if (id == 1L) {
                            started.countDown();
                            await(finish);
                        }
                    };
            try {
                f.submit(1, "b");
                await(started);
                f.submit(2, "c");
                awaitCondition(() -> f.admitted.contains(2L));
                // Two planner slots: R3 must reuse R2's slot while R1 is still held.
                f.submit(3, "d");
                awaitCondition(() -> f.admitted.contains(3L));
                assertEquals(List.of(2L, 3L), f.admissionOrder);
                finish.countDown();
                awaitCondition(() -> f.admitted.contains(1L));
                assertEquals(List.of(2L, 3L, 1L), f.admissionOrder);
            } finally {
                finish.countDown();
            }
        }
    }

    @Test
    void awakenedOlderRequestUsesFreeSlotWithoutInvalidatingRunningPlan() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch finish = new CountDownLatch(1);
            AtomicInteger newerSelections = new AtomicInteger();
            f.onSelection =
                    id -> {
                        if (id == 2L) {
                            newerSelections.incrementAndGet();
                            started.countDown();
                            await(finish);
                        }
                    };
            try {
                f.submit(1, "a");
                f.submit(2, "b");
                await(started);
                f.aSlots.set(1);
                f.release("a");
                awaitCondition(() -> f.admitted.contains(1L));
                assertFalse(f.admitted.contains(2L));
                finish.countDown();
                awaitCondition(() -> f.admitted.size() == 2);
                assertEquals(1, newerSelections.get(), "a wakeup must not discard running work");
            } finally {
                finish.countDown();
            }
        }
    }

    @Test
    void cancelledPlannerKeepsItsSlotUntilItsResourcesAreClosed() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch started = new CountDownLatch(2);
            CountDownLatch finish = new CountDownLatch(1);
            CountDownLatch thirdStarted = new CountDownLatch(1);
            f.onSelection =
                    id -> {
                        if (id <= 2L) {
                            started.countDown();
                            await(finish);
                        } else {
                            thirdStarted.countDown();
                        }
                    };
            try {
                f.submit(1, "b");
                f.submit(2, "c");
                await(started);
                f.requests.get(1L).cancel(false);
                ((QueuedRequestScheduler) f.scheduler).removeRequest(f.contexts.get(1L));
                f.submit(3, "d");
                assertFalse(
                        thirdStarted.await(100, TimeUnit.MILLISECONDS),
                        "cancellation must not allow unbounded outstanding planners");
                finish.countDown();
                awaitCondition(() -> f.admitted.contains(3L));
                assertFalse(f.admitted.contains(1L));

                verify(
                                RequestProtocolTestSupport.placementScheduler(f.mutations.get(1L)),
                                timeout(1000))
                        .finishSelectionWithoutCommit(
                                org.mockito.ArgumentMatchers.same(f.mutations.get(1L)),
                                org.mockito.ArgumentMatchers.isNull());
            } finally {
                finish.countDown();
            }
        }
    }

    @Test
    void shutdownClosesLatePlanWithoutAdmittingIt() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch finish = new CountDownLatch(1);
            f.onSelection =
                    id -> {
                        started.countDown();
                        await(finish);
                    };
            try {
                f.submit(1, "b");
                await(started);
                Thread closer = new Thread(() -> RequestProtocolTestSupport.close(f.scheduler));
                closer.start();
                awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
                finish.countDown();
                awaitCondition(() -> f.routes.containsKey(1L));

                verify(
                                RequestProtocolTestSupport.placementScheduler(f.mutations.get(1L)),
                                timeout(1000))
                        .finishSelectionWithoutCommit(
                                org.mockito.ArgumentMatchers.same(f.mutations.get(1L)),
                                org.mockito.ArgumentMatchers.isNull());
                assertTrue(f.admitted.isEmpty());
                closer.join(2000);
                assertFalse(closer.isAlive());
            } finally {
                finish.countDown();
            }
        }
    }

    @Test
    void freeSlotGoesToHighestPriorityThenFifoWithoutInterruptingRunningPlans() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            CountDownLatch started = new CountDownLatch(2);
            CountDownLatch finishFirst = new CountDownLatch(1);
            CountDownLatch finishSecond = new CountDownLatch(1);
            f.onSelection =
                    id -> {
                        if (id <= 2) {
                            started.countDown();
                            await(id == 1 ? finishFirst : finishSecond);
                        }
                    };
            try {
                f.submit(1, "b", 10);
                f.submit(2, "b", 10);
                await(started);
                f.submit(3, "b", 10);
                f.submit(4, "b", 90);
                f.submit(5, "b", 90);
                finishFirst.countDown();
                awaitCondition(() -> f.admitted.size() == 4);
                assertEquals(List.of(1L, 4L, 5L, 3L), f.admissionOrder);
                assertFalse(f.admitted.contains(2L));
            } finally {
                finishFirst.countDown();
                finishSecond.countDown();
            }
        }
    }

    @Test
    void requestSpecificDecodeFailureDoesNotBlockSmallerRequestsInTheSameGroup() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.aSlots.set(1);
            f.decodeBlocked.add(1L);
            f.submit(1, "a");
            awaitCondition(() -> f.selected.contains(1L));
            f.submit(2, "a");
            awaitCondition(() -> f.admitted.contains(2L));
            assertFalse(f.admitted.contains(1L));
        }
    }

    @Test
    void queuedPreemptionIsNotBlockedByOrdinarySeatExhaustion() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, true)) {
            f.submit(1, "a");
            awaitCondition(() -> f.admitted.contains(1L));
        }
    }

    @Test
    void admissionFailureDoesNotFenceSmallerRequestOnTheSameWorker() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.aSlots.set(1);
            f.admissionBlocked.add(1L);
            f.submit(1, "a");
            f.submit(2, "a");
            awaitCondition(
                    () ->
                            f.admitted.contains(2L)
                                    && RequestProtocolTestSupport.queuedCount(f.scheduler) == 1);
            assertFalse(f.admitted.contains(1L));
            assertEquals(1, RequestProtocolTestSupport.queuedCount(f.scheduler));
        }
    }

    @Test
    void returningPreemptionSlotRetriesWaiterWithoutWorkerCapacityEdge() throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            var pending =
                    new ConcurrentHashMap<
                            Long, CompletableFuture<org.flexlb.balance.schedule.RequestResult>>();
            var attempts = new ConcurrentHashMap<Long, AtomicInteger>();
            var decode = org.flexlb.balance.schedule.SchedulerTestSupport.mockDecodeEndpoint();
            f.onAdmission =
                    id -> {
                        var route = f.routes.get(id);
                        when(route.decodeEndpoint()).thenReturn(decode);
                        when(decode.registrationState(any()))
                                .thenReturn(DecodeResources.RegistrationState.UNCONFIRMED);
                        when(route.requestId()).thenReturn(String.valueOf(id));
                        if (attempts.computeIfAbsent(id, ignored -> new AtomicInteger())
                                        .incrementAndGet()
                                == 1) {
                            f.admissionBlocked.add(id);
                        } else {
                            f.admissionBlocked.remove(id);
                        }
                    };
            when(f.eviction.tryPreempt(any(), eq(decode)))
                    .thenAnswer(
                            call -> {
                                RequestContext context = call.getArgument(0);
                                var operation =
                                        new CompletableFuture<
                                                org.flexlb.balance.schedule.RequestResult>();
                                pending.put(Long.parseLong(context.getRequestId()), operation);
                                return operation;
                            });
            try {
                f.submit(1001L, "a");
                awaitCondition(() -> pending.containsKey(1001L));
                f.submit(1002L, "a");
                awaitCondition(() -> pending.containsKey(1002L));
                f.submit(1003L, "a");
                awaitCondition(() -> attempts.containsKey(1003L));
                assertFalse(
                        pending.containsKey(1003L), "preemption operations must remain bounded");
                pending.get(1001L)
                        .complete(
                                new org.flexlb.balance.schedule.RequestResult(
                                        null, false, "no victim"));
                awaitCondition(() -> f.admitted.contains(1003L));
                assertEquals(
                        2,
                        attempts.get(1003L).get(),
                        "local slot return grants a retry without a capacity edge");
            } finally {
                pending.values()
                        .forEach(
                                operation ->
                                        operation.complete(
                                                new org.flexlb.balance.schedule.RequestResult(
                                                        null, false, "test finished")));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preemptionCompletionResumesOnPlannerAndCommitsWithoutSelectionExit(boolean immediate)
            throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            long id = 1101L;
            DecodeEndpoint decode = f.blockDecodePreemption(id);
            var reservation = new DecodeEndpoint.RequestRegistration(1L, String.valueOf(id));
            var result = new RequestResult(reservation, false, "committed");
            var reply = new CompletableFuture<RequestResult>();
            CountDownLatch invoked = new CountDownLatch(1);
            AtomicReference<Thread> selector = new AtomicReference<>();
            AtomicReference<Thread> commit = new AtomicReference<>();
            LongConsumer prepare = f.onSelection;
            f.onSelection =
                    requestId -> {
                        selector.set(Thread.currentThread());
                        prepare.accept(requestId);
                    };
            LongConsumer admission = f.onAdmission;
            f.onAdmission =
                    requestId -> {
                        commit.set(Thread.currentThread());
                        admission.accept(requestId);
                    };
            when(f.eviction.tryPreempt(any(), eq(decode)))
                    .thenAnswer(
                            call -> {
                                invoked.countDown();
                                return immediate
                                        ? CompletableFuture.completedFuture(result)
                                        : reply;
                            });
            try {
                f.submit(id, "a");
                await(invoked);
                if (!immediate) {
                    Thread remote =
                            new Thread(() -> reply.complete(result), "test-preemption-reply");
                    remote.start();
                    remote.join(2_000);
                    assertFalse(remote.isAlive());
                }
                awaitCondition(() -> f.admitted.contains(id));
                assertTrue(selector.get().getName().startsWith("flexlb-global-planner-"));
                assertTrue(commit.get().getName().startsWith("flexlb-global-planner-"));
                assertFalse(selector.get().isVirtual());
                assertTrue(selector.get().isDaemon());
                assertEquals(List.of(id), f.admissionOrder);

                verify(RequestProtocolTestSupport.placementScheduler(f.mutations.get(id)), never())
                        .finishSelectionWithoutCommit(
                                org.mockito.ArgumentMatchers.same(f.mutations.get(id)),
                                org.mockito.ArgumentMatchers.isNull());
                assertEquals(0L, f.contexts.get(id).snapshot().selectionId());
                assertEquals(
                        RequestContext.ResponseKind.ACK,
                        f.contexts.get(id).snapshot().responseForFuture().responseKind());
                verify(f.router)
                        .select(eq(f.contexts.get(id)), eq("a"), eq(decode), eq(reservation));
                verify(decode, never()).releaseRequest(any());
            } finally {
                reply.complete(result);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"empty", "notCommitted", "exception"})
    void unsuccessfulPreemptionTerminatesAndReturnsItsOperationQuota(String outcome)
            throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            long id = 1102L;
            DecodeEndpoint decode = f.blockDecodePreemption(id);
            var reply = new CompletableFuture<RequestResult>();
            CountDownLatch invoked = new CountDownLatch(1);
            when(f.eviction.tryPreempt(any(), eq(decode)))
                    .thenAnswer(
                            call -> {
                                invoked.countDown();
                                return reply;
                            });
            try {
                f.submit(id, "a");
                await(invoked);
                switch (outcome) {
                    case "empty" -> reply.complete(null);
                    case "notCommitted" ->
                            reply.complete(new RequestResult(null, false, "no victim"));
                    case "exception" ->
                            reply.completeExceptionally(
                                    new IllegalStateException("control failed"));
                    default -> throw new AssertionError(outcome);
                }
                var response = ArgumentCaptor.forClass(Response.class);
                verify(
                                RequestProtocolTestSupport.placementScheduler(f.mutations.get(id)),
                                timeout(1000))
                        .finishSelectionWithoutCommit(
                                org.mockito.ArgumentMatchers.same(f.mutations.get(id)),
                                response.capture());
                assertFalse(response.getValue().isSuccess());

                verify(RequestProtocolTestSupport.placementScheduler(f.mutations.get(id)), never())
                        .finishSelectionWithoutCommit(
                                org.mockito.ArgumentMatchers.same(f.mutations.get(id)),
                                org.mockito.ArgumentMatchers.isNull());
                assertFalse(f.admitted.contains(id));
                awaitCondition(() -> !f.hasPendingPreemptions());
                verify(decode, never()).registrationState(any());
                verify(decode, never()).releaseRequest(any());
                f.submit(1103L, "b");
                awaitCondition(() -> f.admitted.contains(1103L));
                awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
            } finally {
                reply.complete(new RequestResult(null, false, "test finished"));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateCommittedPreemptionAfterCancelOrCloseRollsBackExactReservation(boolean shutdown)
            throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            long id = 1104L;
            DecodeEndpoint decode = f.blockDecodePreemption(id);
            var reservation = new DecodeEndpoint.RequestRegistration(3L, String.valueOf(id));
            var reply = new CompletableFuture<RequestResult>();
            CountDownLatch invoked = new CountDownLatch(1);
            when(f.eviction.tryPreempt(any(), eq(decode)))
                    .thenAnswer(
                            call -> {
                                invoked.countDown();
                                return reply;
                            });
            Thread closer = null;
            try {
                f.submit(id, "a");
                await(invoked);
                if (shutdown) {
                    closer =
                            new Thread(
                                    () -> RequestProtocolTestSupport.close(f.scheduler),
                                    "test-scheduler-close");
                    closer.start();
                    awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
                    assertTrue(
                            closer.isAlive(),
                            "close must retain the pending preemption responsibility");
                } else {
                    assertTrue(f.requests.get(id).cancel(false));
                    ((QueuedRequestScheduler) f.scheduler).removeRequest(f.contexts.get(id));
                    awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
                    f.submit(1105L, "b");
                    awaitCondition(() -> f.admitted.contains(1105L));
                }
                reply.complete(new RequestResult(reservation, false, "late committed result"));
                verify(decode, timeout(1000).times(1)).rollbackSelection(reservation);

                verify(
                                RequestProtocolTestSupport.placementScheduler(f.mutations.get(id)),
                                timeout(1000).times(1))
                        .finishSelectionWithoutCommit(
                                org.mockito.ArgumentMatchers.same(f.mutations.get(id)),
                                org.mockito.ArgumentMatchers.isNull());
                verify(decode).registrationState(reservation);
                assertFalse(f.admitted.contains(id));
                if (closer != null) {
                    closer.join(2_000);
                    assertFalse(closer.isAlive());
                }
            } finally {
                reply.complete(new RequestResult(reservation, false, "test finished"));
                if (closer != null) {
                    closer.join(2_000);
                }
            }
        }
    }

    @Test
    void closeWaitsForLatePreemptionReservationReleaseAfterItsFutureCompletes() throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            long id = 1106L;
            DecodeEndpoint decode = f.blockDecodePreemption(id);
            var reservation = new DecodeEndpoint.RequestRegistration(3L, String.valueOf(id));
            var reply = new CompletableFuture<RequestResult>();
            CountDownLatch invoked = new CountDownLatch(1);
            CountDownLatch releasing = new CountDownLatch(1);
            CountDownLatch allowRelease = new CountDownLatch(1);
            when(f.eviction.tryPreempt(any(), eq(decode)))
                    .thenAnswer(
                            call -> {
                                invoked.countDown();
                                return reply;
                            });
            doAnswer(
                            call -> {
                                releasing.countDown();
                                await(allowRelease);
                                return null;
                            })
                    .when(decode)
                    .rollbackSelection(reservation);
            Thread closer =
                    new Thread(
                            () -> RequestProtocolTestSupport.close(f.scheduler),
                            "test-close-during-release");
            Thread completer =
                    new Thread(
                            () ->
                                    reply.complete(
                                            new RequestResult(
                                                    reservation, false, "late committed result")),
                            "test-late-preemption-result");
            try {
                f.submit(id, "a");
                await(invoked);
                closer.start();
                awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
                completer.start();
                await(releasing);
                assertTrue(reply.isDone());
                assertTrue(
                        closer.isAlive(),
                        "close must wait for resource release, not just the result Future");
                assertFalse(f.admitted.contains(id));
            } finally {
                allowRelease.countDown();
                reply.complete(new RequestResult(reservation, false, "test finished"));
                closer.join(2_000);
                completer.join(2_000);
            }
            assertFalse(closer.isAlive());
            verify(decode).rollbackSelection(reservation);
        }
    }

    @Test
    void selectorCapacityMissCannotStartPreemptionWithoutAnEndpoint() throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            f.decodeBlocked.add(1106L);
            f.submit(1106L, "a");
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(f.scheduler, 1);
            f.submit(1107L, "b");
            awaitCondition(() -> f.admitted.contains(1107L));
            verify(f.eviction, never()).tryPreempt(any(), any());
            assertFalse(f.admitted.contains(1106L));
            f.decodeBlocked.remove(1106L);
            f.release("a");
            awaitCondition(() -> f.admitted.contains(1106L));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void stalePublicationBlockerReplansImmediatelyWithFreshSelection(boolean priority)
            throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, priority)) {
            var old = new AtomicReference<RequestResult>();
            var attempts = new AtomicInteger();
            f.onAdmission =
                    id -> {
                        if (attempts.getAndIncrement() == 0) {
                            old.set(f.routes.get(id));
                            f.stalePublications.add(id);
                        } else {
                            f.admissionBlocked.remove(id);
                        }
                    };
            f.submit(1201L, "b");
            awaitCondition(() -> f.admitted.contains(1201L));
            assertEquals(2, attempts.get());
            org.junit.jupiter.api.Assertions.assertNotSame(old.get(), f.routes.get(1201L));

            verify(f.eviction, never()).tryPreempt(any(), any());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void arrivalsAndUnrelatedCapacityCannotRetryAParkedRequest(boolean priority) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, priority)) {
            f.aSlots.set(1);
            f.decodeBlocked.add(1202L);
            var attempts = new AtomicInteger();
            f.onSelection =
                    id -> {
                        if (id == 1202L) {
                            attempts.incrementAndGet();
                        }
                    };
            f.submit(1202L, "a");
            RequestProtocolTestSupport.awaitGlobalCapacityWaiters(f.scheduler, 1);
            for (int index = 0; index < 10; index++) {
                f.release("unrelated");
            }
            f.submit(1203L, "b");
            awaitCondition(() -> f.admitted.contains(1203L));
            assertEquals(1, attempts.get());
            assertFalse(f.admitted.contains(1202L));
            f.decodeBlocked.remove(1202L);
            f.aSlots.set(1);
            CapacityNotifierTestSupport.notifyChanged(
                    f.availability, CapacityKey.exact(RoleType.DECODE, "a", "d:8000"));
            awaitCondition(() -> f.admitted.contains(1202L));
            assertEquals(2, attempts.get());
        }
    }

    /**
     * Controls placement and preemption outcomes while the real global planner owns queue progress and task slots.
     */
    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = twoPlannerConfig();
        private final CapacityNotifier availability = new CapacityNotifier();
        private final AtomicInteger aSlots = new AtomicInteger();
        private final Map<Long, RequestResult> routes = new ConcurrentHashMap<>();
        private final Map<Long, SelectionAttempt> mutations = new ConcurrentHashMap<>();
        private final Map<Long, String> groups = new ConcurrentHashMap<>();
        private final Map<Long, RequestContext> contexts = new ConcurrentHashMap<>();
        private final Map<Long, CompletableFuture<Response>> requests = new ConcurrentHashMap<>();
        private final Set<Long> selected = ConcurrentHashMap.newKeySet();
        private final Set<Long> admitted = ConcurrentHashMap.newKeySet();
        private final List<Long> admissionOrder = new CopyOnWriteArrayList<>();
        private final Set<Long> admissionBlocked = ConcurrentHashMap.newKeySet();
        private final Set<Long> stalePublications = ConcurrentHashMap.newKeySet();
        private final Map<CapacityKey, WorkerEndpoint> blockers = new ConcurrentHashMap<>();
        private final Set<Long> decodeBlocked = ConcurrentHashMap.newKeySet();

        private volatile LongConsumer onAdmission = ignored -> {};
        private volatile LongConsumer onSelection = ignored -> {};

        private final AbstractRequestScheduler scheduler;
        private final AbstractRequestScheduler lifecycle;
        private final RoleType role;
        private final RequestRouteSelector router = mock(RequestRouteSelector.class);
        private final DecodePreemption eviction =
                org.flexlb.balance.schedule.DecodePreemptionTestSupport.mock();

        private Fixture(RoleType role) {
            this(role, false);
        }

        private Fixture(RoleType role, boolean preempt) {
            this.role = role;
            SchedulingTestConfig.useFifoQueue(config);
            if (preempt) {
                SchedulingTestConfig.usePriorityQueue(config);
                SchedulingTestConfig.allowVictim(config, PreemptionStage.DECODE_ENGINE_OWNED);
            }
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            when(router.resolvePolicyGroup(any()))
                    .thenAnswer(
                            i ->
                                    groups.get(
                                            Long.parseLong(
                                                    ((RequestContext) i.getArgument(0))
                                                            .getRequestId())));

            PrefillEndpoint a = endpoint("a");
            lifecycle = RequestProtocolTestSupport.schedulerMock();
            when(lifecycle.submit(any(), org.mockito.ArgumentMatchers.any(Runnable.class)))
                    .thenAnswer(
                            i -> {
                                RequestContext context = i.getArgument(0);
                                CompletableFuture<Response> future = new CompletableFuture<>();
                                requests.put(Long.parseLong(context.getRequestId()), future);
                                contexts.put(Long.parseLong(context.getRequestId()), context);
                                context.setFuture(future);
                                if (!future.isDone())
                                    ((AbstractRequestScheduler) i.getMock())
                                            .scheduleRequest(i.getArgument(0));
                                return future;
                            });
            when(lifecycle.tryBeginRouteSelection(any(RequestContext.class)))
                    .thenAnswer(
                            i -> {
                                SelectionAttempt mutation =
                                        RequestProtocolTestSupport.admissionHandle(
                                                i.getArgument(0, RequestContext.class));
                                mutations.put(
                                        Long.parseLong(
                                                i.getArgument(0, RequestContext.class)
                                                        .getRequestId()),
                                        mutation);
                                return mutation;
                            });
            org.mockito.Mockito.doAnswer(
                            commit -> {
                                RequestResult selectedRoute = commit.getArgument(0);
                                RequestContext context = selectedRoute.ctx();
                                long id = Long.parseLong(context.getRequestId());
                                if (!preempt || role != RoleType.DECODE) onAdmission.accept(id);
                                if (stalePublications.remove(id)) return RequestResult.retry();
                                if (admissionBlocked.contains(id)) {
                                    return RequestResult.blocked(
                                            CapacityKey.exact(role, "a", "a:8000"));
                                }
                                admissionOrder.add(id);
                                admitted.add(id);
                                ServerStatus status = new ServerStatus();
                                status.setRole(RoleType.PREFILL);
                                status.setRequestId(context.getRequestId());
                                status.setSuccess(true);
                                RequestResult item =
                                        RequestProtocolTestSupport.mockSelectedRoute(
                                                context,
                                                List.of(
                                                        SelectionTestSupport.metadata(
                                                                a, status, 0L)));
                                return RequestProtocolTestSupport.published(selectedRoute, item);
                            })
                    .when(RequestProtocolTestSupport.publication(lifecycle))
                    .publishSelectedRoute(any());
            when(router.select(any(), nullable(String.class)))
                    .thenAnswer(
                            i -> {
                                RequestContext context = i.getArgument(0);
                                long id = Long.parseLong(context.getRequestId());
                                selected.add(id);
                                onSelection.accept(id);
                                if ("a".equals(groups.get(id)) && aSlots.get() == 0 && !preempt) {
                                    return RequestResult.blocked(new CapacityKey(role, "a", null));
                                }
                                if (decodeBlocked.contains(id)) {
                                    return RequestResult.blocked(
                                            new CapacityKey(RoleType.DECODE, groups.get(id), null));
                                }
                                RequestResult route =
                                        RequestProtocolTestSupport.mockSelectedRoute(context);
                                when(route.ctx()).thenReturn(context);
                                routes.put(id, route);
                                if (preempt && role == RoleType.DECODE) {
                                    onAdmission.accept(id);
                                    if (admissionBlocked.contains(id)) {
                                        var blocker = CapacityKey.exact(role, "a", "a:8000");
                                        if (route.decodeEndpoint() != null)
                                            blockers.put(blocker, route.decodeEndpoint());
                                        return RequestResult.blocked(blocker);
                                    }
                                }
                                return route;
                            });
            when(router.blockedEndpoint(any()))
                    .thenAnswer(call -> blockers.get(call.getArgument(0)));
            when(router.select(any(), nullable(String.class), any(), any()))
                    .thenAnswer(
                            call -> {
                                RequestContext context = call.getArgument(0);
                                long id = Long.parseLong(context.getRequestId());
                                RequestResult route =
                                        RequestProtocolTestSupport.mockSelectedRoute(context);
                                when(route.ctx()).thenReturn(context);
                                when(route.requestId()).thenReturn(String.valueOf(id));
                                when(route.decodeEndpoint()).thenReturn(call.getArgument(2));
                                when(route.decodeRegistration()).thenReturn(call.getArgument(3));
                                routes.put(id, route);
                                onAdmission.accept(id);
                                return route;
                            });
            scheduler =
                    RequestProtocolTestSupport.injectPlacementDependencies(
                            lifecycle,
                            service,
                            router,
                            mock(DeliveryMetricsReporter.class),
                            eviction,
                            availability);
        }

        public void close() {
            requests.values().forEach(future -> future.complete(new Response()));
            RequestProtocolTestSupport.close(scheduler);
        }

        private static FlexlbConfig twoPlannerConfig() {
            FlexlbConfig config = spy(SchedulingTestConfig.newConfig());
            var runtime = spy(config.getInternalRuntime());
            when(runtime.getQueuePlannerThreads()).thenReturn(2);
            when(config.getInternalRuntime()).thenReturn(runtime);
            return config;
        }

        private DecodeEndpoint blockDecodePreemption(long requestId) {
            DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
            AtomicInteger attempts = new AtomicInteger();
            onAdmission =
                    id -> {
                        if (id != requestId) {
                            return;
                        }
                        RequestResult route = routes.get(id);
                        when(route.decodeEndpoint()).thenReturn(decode);
                        when(decode.registrationState(any()))
                                .thenReturn(DecodeResources.RegistrationState.UNCONFIRMED);
                        when(route.requestId()).thenReturn(String.valueOf(id));
                        if (attempts.getAndIncrement() == 0) {
                            admissionBlocked.add(id);
                        } else {
                            admissionBlocked.remove(id);
                        }
                    };
            return decode;
        }

        private void submit(long id, String group) {
            submit(id, group, 50);
        }

        private void submit(long id, String group, int priority) {
            groups.put(id, group);
            RequestContext context =
                    org.mockito.Mockito.spy(RequestProtocolTestSupport.context(config, id));
            context.setSchedulingMetadata(
                    org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                            priority, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
            SchedulerTestServices.entry(scheduler).submit(context);
        }

        private boolean hasPendingPreemptions() {
            var queueLock = (ReentrantLock) ReflectionTestUtils.getField(scheduler, "lock");
            queueLock.lock();
            try {
                return !(((SchedulingRequests)
                                        ReflectionTestUtils.getField(
                                                scheduler, "schedulingRequests"))
                                .preemptionCount()
                        == 0);
            } finally {
                queueLock.unlock();
            }
        }

        private void release(String group) {
            CapacityNotifierTestSupport.notifyChanged(
                    availability, CapacityKey.exact(role, group, group + ":8000"));
        }

        private static PrefillEndpoint endpoint(String group) {
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            WorkerStatus status = mock(WorkerStatus.class);
            WorkerStatus.TopologySnapshot topology = mock(WorkerStatus.TopologySnapshot.class);
            when(status.topologySnapshot()).thenReturn(topology);
            when(topology.group()).thenReturn(group);
            when(endpoint.getStatus()).thenReturn(status);
            return endpoint;
        }
    }
}
