package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.List;
import java.util.Map;
import java.util.Queue;
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

import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.await;
import static org.flexlb.balance.scheduler.RequestProtocolTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GlobalQueueProgressTest {

    @Test
    void releasingBlockedPlanPublishesCapacityBeforeParking() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            AtomicInteger attempts = new AtomicInteger();
            AtomicInteger closes = new AtomicInteger();
            f.onSelection = id -> {
                if (attempts.get() > 0) {
                    assertEquals(1, closes.get(), "retry cannot begin before the old route closes");
                }
            };
            f.onAdmission = id -> {
                if (attempts.getAndIncrement() == 0) {
                    f.admissionBlocked.add(id);
                    RequestRoute route = f.routes.get(id);
                    when(route.blockedEndpointIfCurrent(any())).thenReturn(mock(PrefillEndpoint.class));
                    doAnswer(invocation -> {
                        closes.incrementAndGet();
                        f.admissionBlocked.remove(id);
                        f.release("a");
                        return null;
                    }).when(route).close();
                }
            };
            f.submit(993L, "b");
            awaitCondition(() -> f.admitted.contains(993L));
            assertEquals(2, attempts.get(), "the cleanup edge must grant a retry without another event");
            assertEquals(1, closes.get());
        }
    }

    @Test
    void parkedPlanRetainsSelectionDiagnosticsAfterClosing() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            Map<String, Object> diagnostics = Map.of("reason", "no eligible worker");
            doReturn(PlacementResult.blocked(new PlacementKey(RoleType.PREFILL, "a", null), null, diagnostics))
                    .when(f.router).select(any(), nullable(String.class));
            f.submit(994L, "a");
            QueuedRequestScheduler queue = (QueuedRequestScheduler) f.scheduler;
            awaitCondition(() -> diagnostics.equals(queue.getLatestQueueWaitSnapshot().get("decision")));
            verify(f.mutations.get(994L)).finish();
        }
    }

    @Test
    void handleCloseFailureReportsFailureAndReleasesPlanningSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.onAdmission = id -> {
                if (id == 995L) {
                    doThrow(new IllegalStateException("handle close failed")).when(f.mutations.get(id)).finish();
                }
            };
            f.submit(995L, "b");
            var response = ArgumentCaptor.forClass(Response.class);
            verify(f.lifecycle, timeout(5_000)).publishDecisionResponseAsync(
                    eq(995L), eq(f.requests.get(995L)), response.capture());
            assertEquals("Placement failed: handle close failed", response.getValue().getErrorMessage());
            verify(f.mutations.get(995L)).finish();
            f.submit(996L, "b");
            awaitCondition(() -> f.admitted.contains(996L));
        }
    }

    @Test
    void routeRollbackFailureIsReportedWithoutStrandingThePlanningSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            var failure = new IllegalStateException("route rollback failed");
            f.onAdmission = id -> {
                if (id == 997L) { doThrow(failure).when(f.routes.get(id)).close(); }
            };
            f.submit(997L, "b");
            verify(f.lifecycle, timeout(1_000)).recordFailure(failure);
            verify(f.mutations.get(997L), timeout(1_000)).finish();
            f.submit(998L, "b");
            awaitCondition(() -> f.admitted.contains(998L));
        }
    }

    @Test
    void missingPlacementResultDoesNotStrandTheAdmissionOwner() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            doReturn(null).when(f.router).select(any(), nullable(String.class));
            f.submit(999L, "a");
            verify(f.lifecycle, timeout(1_000)).publishDecisionResponseAsync(
                    eq(999L), eq(f.requests.get(999L)), any());
            verify(f.mutations.get(999L)).finish();
        }
    }

    @Test
    void failedPlanPreparationClosesSelectedRouteAndPreservesItsCause() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            f.aSlots.set(1);
            var primary = new IllegalStateException("diagnostics failed");
            var cleanup = new IllegalStateException("handle close failed");
            var finishingThread = new java.util.concurrent.atomic.AtomicReference<Thread>();
            f.onSelection = id -> {
                AdmissionHandle handle = f.mutations.get(id);
                doThrow(primary).when(handle).recordDiagnostics(nullable(Map.class));
                doAnswer(ignored -> {
                    finishingThread.set(Thread.currentThread());
                    throw cleanup;
                }).when(handle).finish();
            };
            f.submit(992L, "a");
            var response = ArgumentCaptor.forClass(Response.class);
            verify(f.lifecycle, timeout(5_000)).publishDecisionResponseAsync(
                    eq(992L), eq(f.requests.get(992L)), response.capture());
            assertEquals("Placement failed: diagnostics failed", response.getValue().getErrorMessage());
            assertEquals(List.of(cleanup), List.of(primary.getSuppressed()));
            assertSame(ReflectionTestUtils.getField(f.scheduler, "decisionThread"), finishingThread.get(),
                    "planning failures must return to the decision owner before admission completion");
            verify(f.routes.get(992L)).close();
            verify(f.mutations.get(992L)).finish();
            assertFalse(f.admitted.contains(992L));
        }
    }

    @Test
    void nonTerminalControlFactRestoresTheExactGlobalQueueEntry() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            doAnswer(call -> {
                RequestContext context = call.getArgument(0);
                CompletableFuture<Response> future = (CompletableFuture<Response>) call.callRealMethod();
                f.contexts.put(context.getRequestId(), context);
                f.requests.put(context.getRequestId(), future);
                return future;
            }).when(f.lifecycle).register(any(), any());
            f.submit(991L, "a");
            awaitCondition(() -> f.selected.contains(991L));
            QueuedRequestScheduler queue = (QueuedRequestScheduler)
                    f.scheduler;
            RequestContext context = f.contexts.get(991L);
            queue.signalControl(context);
            verify((QueuedRequestScheduler) f.lifecycle, timeout(5_000)).onGlobalControl(context);

            f.aSlots.set(1);
            f.release("a");
            awaitCondition(() -> f.admitted.contains(991L));
        }
    }

    @Test
    void completedBacklogDoesNotDelayRefillingAReleasedSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch firstPlanning = new CountDownLatch(1);
            CountDownLatch finishFirstPlanning = new CountDownLatch(1);
            CountDownLatch firstCommitting = new CountDownLatch(1);
            CountDownLatch secondCommitting = new CountDownLatch(1);
            CountDownLatch finishFirst = new CountDownLatch(1);
            CountDownLatch finishSecond = new CountDownLatch(1);
            CountDownLatch thirdPlanning = new CountDownLatch(1);
            f.onAdmission = id -> {
                if (id == 1L) {
                    firstCommitting.countDown();
                    await(finishFirst);
                } else if (id == 2L) {
                    secondCommitting.countDown();
                    await(finishSecond);
                }
            };
            f.onSelection = id -> {
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
                awaitCondition(f::hasBufferedPlan);
                f.submit(3, "d");
                finishFirst.countDown();
                await(secondCommitting);
                assertTrue(thirdPlanning.await(1, TimeUnit.SECONDS),
                        "R3 must reuse R1's slot before the buffered R2 commit finishes");
            } finally {
                finishFirstPlanning.countDown();
                finishFirst.countDown();
                finishSecond.countDown();
            }
        }
    }

    @Test
    void slowPlannerDoesNotBlockOtherResultsOrRefillingItsPeersSlot() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch started = new CountDownLatch(1);
            CountDownLatch finish = new CountDownLatch(1);
            f.onSelection = id -> {
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
            f.onSelection = id -> {
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
            f.onSelection = id -> {
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
                f.submit(3, "d");
                assertFalse(thirdStarted.await(100, TimeUnit.MILLISECONDS),
                        "cancellation must not allow unbounded outstanding planners");
                finish.countDown();
                awaitCondition(() -> f.admitted.contains(3L));
                assertFalse(f.admitted.contains(1L));
                verify(f.routes.get(1L), timeout(1000)).close();
                verify(f.mutations.get(1L), timeout(1000)).finish();
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
            f.onSelection = id -> {
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
                verify(f.routes.get(1L), timeout(1000)).close();
                verify(f.mutations.get(1L), timeout(1000)).finish();
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
            f.onSelection = id -> {
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
            awaitCondition(() -> f.admitted.contains(2L)
                    && RequestProtocolTestSupport.queuedCount(f.scheduler) == 1);
            assertFalse(f.admitted.contains(1L));
            assertEquals(1, RequestProtocolTestSupport.queuedCount(f.scheduler));
        }
    }

    @Test
    void returningPreemptionSlotRetriesWaiterWithoutWorkerCapacityEdge() throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            var pending = new ConcurrentHashMap<Long, CompletableFuture<org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult>>();
            var attempts = new ConcurrentHashMap<Long, AtomicInteger>();
            var decode = mock(org.flexlb.balance.endpoint.DecodeEndpoint.class);
            f.onSelection = id -> ReflectionTestUtils.setField(f.contexts.get(id), "requirements",
                    RequestRequirements.capture(f.contexts.get(id)));
            f.onAdmission = id -> {
                var route = f.routes.get(id);
                when(route.blockedEndpointIfCurrent(any())).thenReturn(decode);
                when(route.decodeEp()).thenReturn(decode);
                when(decode.markQueued(any(), any())).thenReturn(true);
                when(route.requestId()).thenReturn(id);
                if (attempts.computeIfAbsent(id, ignored -> new AtomicInteger()).incrementAndGet() == 1) {
                    f.admissionBlocked.add(id);
                } else { f.admissionBlocked.remove(id); }
            };
            when(f.eviction.tryReclaim(any(), any(), eq(decode))).thenAnswer(call -> {
                RequestContext context = call.getArgument(0);
                var operation = new CompletableFuture<org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult>();
                pending.put(context.getRequestId(), operation);
                return operation;
            });
            try {
                f.submit(1001L, "a");
                awaitCondition(() -> pending.containsKey(1001L));
                f.submit(1002L, "a");
                awaitCondition(() -> pending.containsKey(1002L));
                f.submit(1003L, "a");
                awaitCondition(() -> attempts.containsKey(1003L));
                assertFalse(pending.containsKey(1003L), "preemption operations must remain bounded");
                pending.get(1001L).complete(new org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult(
                        null, false, "no victim"));
                awaitCondition(() -> f.admitted.contains(1003L));
                assertEquals(2, attempts.get(1003L).get(), "local slot return grants a retry without a capacity edge");
            } finally {
                pending.values().forEach(operation -> operation.complete(
                        new org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult(null, false, "test finished")));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void preemptionCompletionCommitsOnDecisionThreadAndClosesExactlyOnce(boolean immediate) throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            long id = 1101L;
            DecodeEndpoint decode = f.blockDecodeAdmission(id);
            var reservation = new DecodeResources.ReservationHandle(1L, id, 7L);
            var result = new PreemptionResult(reservation, false, "committed");
            var reply = new CompletableFuture<PreemptionResult>();
            CountDownLatch invoked = new CountDownLatch(1);
            AtomicReference<Thread> selector = new AtomicReference<>();
            AtomicReference<Thread> commit = new AtomicReference<>();
            LongConsumer prepare = f.onSelection;
            f.onSelection = requestId -> {
                selector.set(Thread.currentThread());
                prepare.accept(requestId);
            };
            LongConsumer admission = f.onAdmission;
            f.onAdmission = requestId -> {
                commit.set(Thread.currentThread());
                admission.accept(requestId);
            };
            when(f.eviction.tryReclaim(any(), any(), eq(decode))).thenAnswer(call -> {
                invoked.countDown();
                return immediate ? CompletableFuture.completedFuture(result) : reply;
            });
            try {
                f.submit(id, "a");
                await(invoked);
                if (!immediate) {
                    Thread remote = new Thread(() -> reply.complete(result), "test-preemption-reply");
                    remote.start();
                    remote.join(2_000);
                    assertFalse(remote.isAlive());
                }
                awaitCondition(() -> f.admitted.contains(id));
                assertTrue(selector.get().getName().startsWith("flexlb-global-planner-"));
                assertEquals("flexlb-global-decision", commit.get().getName());
                assertFalse(selector.get().isVirtual());
                assertTrue(selector.get().isDaemon());
                assertEquals(List.of(id), f.admissionOrder);
                verify(f.routes.get(id), timeout(1000).times(1)).close();
                verify(f.mutations.get(id), timeout(1000).times(1)).finish();
                verify(decode, times(1)).markQueued(any(), eq(reservation));
                verify(decode, never()).release(any(), any());
            } finally {
                reply.complete(result);
            }
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"empty", "notCommitted", "exception"})
    void unsuccessfulPreemptionTerminatesAndReturnsItsOperationQuota(String outcome) throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            long id = 1102L;
            DecodeEndpoint decode = f.blockDecodeAdmission(id);
            var reply = new CompletableFuture<PreemptionResult>();
            CountDownLatch invoked = new CountDownLatch(1);
            when(f.eviction.tryReclaim(any(), any(), eq(decode))).thenAnswer(call -> {
                invoked.countDown();
                return reply;
            });
            try {
                f.submit(id, "a");
                await(invoked);
                switch (outcome) {
                    case "empty" -> reply.complete(null);
                    case "notCommitted" -> reply.complete(new PreemptionResult(null, false, "no victim"));
                    case "exception" -> reply.completeExceptionally(new IllegalStateException("control failed"));
                    default -> throw new AssertionError(outcome);
                }
                var response = ArgumentCaptor.forClass(Response.class);
                verify(f.mutations.get(id), timeout(1000)).terminate(response.capture());
                assertFalse(response.getValue().isSuccess());
                verify(f.routes.get(id), timeout(1000).times(1)).close();
                verify(f.mutations.get(id), timeout(1000).times(1)).finish();
                assertFalse(f.admitted.contains(id));
                awaitCondition(() -> !f.hasPendingPreemptions());
                verify(decode, never()).markQueued(any(), any());
                verify(decode, never()).release(any(), any());
                f.submit(1103L, "b");
                awaitCondition(() -> f.admitted.contains(1103L));
                awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
            } finally {
                reply.complete(new PreemptionResult(null, false, "test finished"));
            }
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void lateCommittedPreemptionAfterCancelOrCloseRollsBackExactReservation(boolean shutdown) throws Exception {
        try (Fixture f = new Fixture(RoleType.DECODE, true)) {
            long id = 1104L;
            DecodeEndpoint decode = f.blockDecodeAdmission(id);
            var reservation = new DecodeResources.ReservationHandle(3L, id, 17L);
            var reply = new CompletableFuture<PreemptionResult>();
            CountDownLatch invoked = new CountDownLatch(1);
            when(f.eviction.tryReclaim(any(), any(), eq(decode))).thenAnswer(call -> {
                invoked.countDown();
                return reply;
            });
            Thread closer = null;
            try {
                f.submit(id, "a");
                await(invoked);
                if (shutdown) {
                    closer = new Thread(() -> RequestProtocolTestSupport.close(f.scheduler), "test-scheduler-close");
                    closer.start();
                    awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
                    assertTrue(closer.isAlive(), "close must retain the pending preemption responsibility");
                } else {
                    assertTrue(f.requests.get(id).cancel(false));
                    awaitCondition(() -> RequestProtocolTestSupport.queuedCount(f.scheduler) == 0);
                    f.submit(1105L, "b");
                    awaitCondition(() -> f.admitted.contains(1105L));
                }
                reply.complete(new PreemptionResult(reservation, false, "late committed result"));
                verify(decode, timeout(1000).times(1)).release(reservation, DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
                verify(f.routes.get(id), timeout(1000).times(1)).close();
                verify(f.mutations.get(id), timeout(1000).times(1)).finish();
                verify(decode, never()).markQueued(any(), any());
                assertFalse(f.admitted.contains(id));
                if (closer != null) {
                    closer.join(2_000);
                    assertFalse(closer.isAlive());
                }
            } finally {
                reply.complete(new PreemptionResult(reservation, false, "test finished"));
                if (closer != null) { closer.join(2_000); }
            }
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
            verify(f.eviction, never()).tryReclaim(any(), any(), any());
            assertFalse(f.admitted.contains(1106L));
            f.decodeBlocked.remove(1106L);
            f.release("a");
            awaitCondition(() -> f.admitted.contains(1106L));
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void stalePublicationBlockerReplansImmediatelyWithFreshSelection(boolean priority) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, priority)) {
            var old = new AtomicReference<RequestRoute>();
            var attempts = new AtomicInteger();
            f.onAdmission = id -> {
                if (attempts.getAndIncrement() == 0) {
                    old.set(f.routes.get(id));
                    f.admissionBlocked.add(id);
                } else { f.admissionBlocked.remove(id); }
            };
            f.submit(1201L, "b");
            awaitCondition(() -> f.admitted.contains(1201L));
            assertEquals(2, attempts.get());
            org.junit.jupiter.api.Assertions.assertNotSame(old.get(), f.routes.get(1201L));
            verify(old.get(), times(1)).close();
            verify(f.eviction, never()).tryReclaim(any(), any(), any());
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void arrivalsAndUnrelatedCapacityCannotRetryAParkedRequest(boolean priority) throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL, priority)) {
            f.aSlots.set(1);
            f.decodeBlocked.add(1202L);
            var attempts = new AtomicInteger();
            f.onSelection = id -> { if (id == 1202L) { attempts.incrementAndGet(); } };
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
            f.availability.changed(PlacementKey.exact(RoleType.DECODE, "a", "d:8000"));
            awaitCondition(() -> f.admitted.contains(1202L));
            assertEquals(2, attempts.get());
        }
    }

    private static final class Fixture implements AutoCloseable {

        private final FlexlbConfig config = twoPlannerConfig();

        private final PlacementAvailability availability = new PlacementAvailability();

        private final AtomicInteger aSlots = new AtomicInteger();

        private final Map<Long, RequestRoute> routes = new ConcurrentHashMap<>();

        private final Map<Long, AdmissionHandle> mutations = new ConcurrentHashMap<>();

        private final Map<Long, String> groups = new ConcurrentHashMap<>();

        private final Map<Long, RequestContext> contexts = new ConcurrentHashMap<>();

        private final Map<Long, CompletableFuture<Response>> requests = new ConcurrentHashMap<>();

        private final Set<Long> selected = ConcurrentHashMap.newKeySet();

        private final Set<Long> admitted = ConcurrentHashMap.newKeySet();

        private final List<Long> admissionOrder = new CopyOnWriteArrayList<>();

        private final Set<Long> admissionBlocked = ConcurrentHashMap.newKeySet();

        private final Set<Long> decodeBlocked = ConcurrentHashMap.newKeySet();

        private volatile LongConsumer onAdmission = ignored -> { };

        private volatile LongConsumer onSelection = ignored -> { };

        private final RequestScheduler scheduler;

        private final AbstractRequestScheduler lifecycle;

        private final RoleType role;

        private final RequestWorkerSelector router = mock(RequestWorkerSelector.class);

        private final DecodeCapacityAcquirer eviction = mock(DecodeCapacityAcquirer.class);

        private static FlexlbConfig twoPlannerConfig() {
            FlexlbConfig config = spy(SchedulingTestConfig.batchConfig());
            var runtime = spy(config.getInternalRuntime());
            when(runtime.getQueuePlannerThreads()).thenReturn(2);
            when(config.getInternalRuntime()).thenReturn(runtime);
            return config;
        }

        private Fixture(RoleType role) {
            this(role, false);
        }

        private Fixture(RoleType role, boolean preempt) {
            this.role = role;
            SchedulingTestConfig.useFifoQueue(config);
            if (preempt) {
                SchedulingTestConfig.usePriorityQueue(config);
                SchedulingTestConfig.allowVictim(config, VictimStage.PREFILL_QUEUED);
            }
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(1);
            ConfigService service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            when(router.resolvePolicyGroup(any())).thenAnswer(i -> groups.get(((RequestContext) i.getArgument(0)).getRequestId()));

            PrefillRoutingEntry a = endpoint("a");
            lifecycle = RequestProtocolTestSupport.schedulerMock();
            when(lifecycle.register(any(), org.mockito.ArgumentMatchers.any())).thenAnswer(i -> {
                RequestContext context = i.getArgument(0);
                CompletableFuture<Response> future = new CompletableFuture<>();
                requests.put(context.getRequestId(), future);
                contexts.put(context.getRequestId(), context);
                context.setFuture(future);
                return future;
            });
            when(lifecycle.claimAdmissionHandle(anyLong(), any())).thenAnswer(i -> {
                AdmissionHandle mutation = mock(AdmissionHandle.class);
                mutations.put(i.getArgument(0), mutation);
                return mutation;
            });
            org.mockito.Mockito.doAnswer(commit -> {
                QueuedRequestScheduler.Plan plan = commit.getArgument(0);
                RequestContext context = RequestProtocolTestSupport.planContext(plan);
                long id = context.getRequestId();
                onAdmission.accept(id);
                if (admissionBlocked.contains(id)) {
                    return PlacementResult.blocked(PlacementKey.exact(role, "a", "a:8000"));
                }
                admissionOrder.add(id);
                admitted.add(id);
                RequestRoute item = mock(RequestRoute.class);
                ServerStatus status = new ServerStatus();
                status.setRole(role);
                when(item.prefill()).thenReturn(status);
                when(item.prefillEp()).thenReturn(a.endpoint());
                return RequestProtocolTestSupport.published(plan, item);
            }).when(RequestProtocolTestSupport.publication(lifecycle)).enqueueRoute(any());
            when(router.select(any(), nullable(String.class))).thenAnswer(i -> {
                RequestContext context = i.getArgument(0);
                long id = context.getRequestId();
                selected.add(id);
                onSelection.accept(id);
                if ("a".equals(groups.get(id)) && aSlots.get() == 0 && !preempt) {
                    return PlacementResult.blocked(new PlacementKey(role, "a", null));
                }
                if (decodeBlocked.contains(id)) {
                    return PlacementResult.blocked(new PlacementKey(RoleType.DECODE, groups.get(id), null));
                }
                RequestRoute route = mock(RequestRoute.class);
                routes.put(id, route);
                return PlacementResult.success(route);
            });
            scheduler = RequestProtocolTestSupport.configure(lifecycle, service, router, mock(DeliveryMetricsReporter.class), eviction, availability);
        }

        private DecodeEndpoint blockDecodeAdmission(long requestId) {
            DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
            AtomicInteger attempts = new AtomicInteger();
            onSelection = id -> SchedulingTestConfig.freezeInputs(contexts.get(id));
            onAdmission = id -> {
                if (id != requestId) { return; }
                RequestRoute route = routes.get(id);
                when(route.blockedEndpointIfCurrent(any())).thenReturn(decode);
                when(route.decodeEp()).thenReturn(decode);
                when(decode.markQueued(any(), any())).thenReturn(true);
                when(route.requestId()).thenReturn(id);
                if (attempts.getAndIncrement() == 0) { admissionBlocked.add(id); }
                else { admissionBlocked.remove(id); }
            };
            return decode;
        }

        private void submit(long id, String group) {
            submit(id, group, 50);
        }

        private void submit(long id, String group, int priority) {
            groups.put(id, group);
            RequestContext context = RequestProtocolTestSupport.context(config, id);
            context.setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(
                    priority, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
            scheduler.submit(context);
        }

        private boolean hasPendingPreemptions() {
            var queueLock = (ReentrantLock) ReflectionTestUtils.getField(scheduler, "lock");
            queueLock.lock();
            try {
                return !((Set<?>) ReflectionTestUtils.getField(scheduler, "pendingPreemptions")).isEmpty();
            } finally {
                queueLock.unlock();
            }
        }

        private boolean hasBufferedPlan() {
            // Observe the handoff under its lock to make the regression independent
            // of thread timing, without adding a hook to production code.
            Object queue = scheduler;
            var lock = (ReentrantLock)
                    ReflectionTestUtils.getField(queue, "lock");
            lock.lock();
            try {
                var completed = (Queue<?>) ReflectionTestUtils
                        .getField(queue, "completedPlans");
                return !completed.isEmpty();
            } finally {
                lock.unlock();
            }
        }

        private void release(String group) {
            availability.changed(PlacementKey.exact(role, group, group + ":8000"));
        }

        public void close() {
            requests.values().forEach(future -> future.complete(new Response()));
            RequestProtocolTestSupport.close(scheduler);
        }

        private static PrefillRoutingEntry endpoint(String group) {
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            WorkerStatus status = mock(WorkerStatus.class);
            WorkerStatus.TopologySnapshot topology = mock(WorkerStatus.TopologySnapshot.class);
            when(status.topologySnapshot()).thenReturn(topology);
            when(topology.group()).thenReturn(group);
            when(endpoint.getStatus()).thenReturn(status);
            return new PrefillRoutingEntry(group + ":8000", endpoint);
        }
    }
}
