package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointRegistry.PrefillRoutingEntry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
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
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongConsumer;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.awaitCondition;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.nullable;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.when;

class GlobalQueueProgressTest {
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
                verify(f.mutations.get(1L), timeout(1000)).close();
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
                Thread closer = new Thread(f.scheduler::closePlacement);
                closer.start();
                awaitCondition(() -> f.scheduler.getQueuedRequestCount() == 0);
                finish.countDown();
                awaitCondition(() -> f.routes.containsKey(1L));
                verify(f.routes.get(1L), timeout(1000)).close();
                verify(f.mutations.get(1L), timeout(1000)).close();
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
            awaitCondition(() -> f.admitted.contains(2L));
            assertFalse(f.admitted.contains(1L));
            assertEquals(1, f.scheduler.getQueuedRequestCount());
        }
    }

    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = twoPlannerConfig();
        private final PlacementAvailability availability = new PlacementAvailability();
        private final AtomicInteger aSlots = new AtomicInteger();
        private final Map<Long, RouteAdmission> routes = new ConcurrentHashMap<>();
        private final Map<Long, AdmissionMutation> mutations = new ConcurrentHashMap<>();
        private final Map<Long, String> groups = new ConcurrentHashMap<>();
        private final Map<Long, CompletableFuture<Response>> requests = new ConcurrentHashMap<>();
        private final Set<Long> selected = ConcurrentHashMap.newKeySet();
        private final Set<Long> admitted = ConcurrentHashMap.newKeySet();
        private final List<Long> admissionOrder = new CopyOnWriteArrayList<>();
        private final Set<Long> admissionBlocked = ConcurrentHashMap.newKeySet();
        private final Set<Long> decodeBlocked = ConcurrentHashMap.newKeySet();
        private volatile LongConsumer onAdmission = ignored -> { };
        private volatile LongConsumer onSelection = ignored -> { };
        private final RequestScheduler scheduler;
        private final RoleType role;

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
            DefaultRouter router = mock(DefaultRouter.class);
            when(router.queueAdmissionRole()).thenReturn(role);
            when(router.resolvePolicyGroup(any())).thenAnswer(i -> groups.get(((BalanceContext) i.getArgument(0)).getRequestId()));
            EndpointRegistry endpoints = mock(EndpointRegistry.class);
            PrefillRoutingEntry a = endpoint("a");
            RequestRegistry lifecycle = mock(RequestRegistry.class);
            when(lifecycle.register(any())).thenAnswer(i -> {
                BalanceContext context = i.getArgument(0);
                CompletableFuture<Response> future = new CompletableFuture<>();
                requests.put(context.getRequestId(), future);
                return future;
            });
            when(lifecycle.claimAdmissionMutation(anyLong(), any())).thenAnswer(i -> {
                AdmissionMutation mutation = mock(AdmissionMutation.class);
                mutations.put(i.getArgument(0), mutation);
                return mutation;
            });
            when(router.select(any(), nullable(String.class))).thenAnswer(i -> {
                BalanceContext context = i.getArgument(0);
                long id = context.getRequestId();
                selected.add(id);
                onSelection.accept(id);
                if ("a".equals(groups.get(id)) && aSlots.get() == 0 && !preempt) {
                    return PlacementResult.blocked(new PlacementKey(role, "a"));
                }
                if (decodeBlocked.contains(id)) {
                    return PlacementResult.blocked(new PlacementKey(RoleType.DECODE, groups.get(id)));
                }
                RouteAdmission route = mock(RouteAdmission.class);
                routes.put(id, route);
                ScheduledRequest item = mock(ScheduledRequest.class);
                ServerStatus status = new ServerStatus();
                status.setRole(role);
                when(item.prefill()).thenReturn(status);
                when(item.prefillEp()).thenReturn(a.endpoint());
                when(route.tryEnqueue(any(), any(), any())).thenAnswer(commit -> {
                    onAdmission.accept(id);
                    if (admissionBlocked.contains(id)) {
                        return PlacementResult.blocked(PlacementKey.exact(role, "a", "a:8000"));
                    }
                    admissionOrder.add(id);
                    admitted.add(id);
                    return PlacementResult.success(item);
                });
                return PlacementResult.success(route);
            });
            scheduler = new RequestScheduler(service, router, endpoints, mock(BatchSchedulerReporter.class),
                    mock(EvictionManager.class), lifecycle, availability);
        }

        private void submit(long id, String group) {
            submit(id, group, 50);
        }

        private void submit(long id, String group, int priority) {
            groups.put(id, group);
            BalanceContext context = RequestLifecycleTestSupport.context(config, id);
            context.setSchedulingMetadata(org.flexlb.dao.SchedulingMetadata.explicit(
                    priority, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
            scheduler.submit(context);
        }

        private boolean hasBufferedPlan() {
            // Observe the handoff under its lock to make the regression independent
            // of thread timing, without adding a hook to production code.
            Object queue = ReflectionTestUtils.getField(scheduler, "globalQueue");
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
            availability.capacityChanged(PlacementKey.exact(role, group, group + ":8000"));
        }

        public void close() {
            requests.values().forEach(future -> future.complete(new Response()));
            scheduler.closePlacement();
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
