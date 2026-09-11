package org.flexlb.balance.scheduler;

import static org.mockito.ArgumentMatchers.eq;


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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
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
import static org.mockito.Mockito.when;

class GlobalQueueProgressTest {
    @Test
    void olderRequestAwakenedDuringPlanningIsRetriedBeforeTheNewerPlanCommits() throws Exception {
        try (Fixture f = new Fixture(RoleType.PREFILL)) {
            CountDownLatch planningStarted = new CountDownLatch(1);
            CountDownLatch finishPlanning = new CountDownLatch(1);
            f.onSelection = id -> {
                if (id == 2L) {
                    planningStarted.countDown();
                    await(finishPlanning);
                }
            };
            try {
                f.submit(1, "a");
                f.submit(2, "b");
                await(planningStarted);
                f.aSlots.set(1);
                f.release("a");
                finishPlanning.countDown();
                awaitCondition(() -> f.admitted.size() == 2);
                assertEquals(List.of(1L, 2L), f.admissionOrder);
            } finally {
                finishPlanning.countDown();
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

    private static final class Fixture implements AutoCloseable {
        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        private final PlacementAvailability availability = new PlacementAvailability();
        private final AtomicInteger aSlots = new AtomicInteger();
        private final Map<Long, String> groups = new ConcurrentHashMap<>();
        private final Map<Long, CompletableFuture<Response>> requests = new ConcurrentHashMap<>();
        private final Set<Long> selected = ConcurrentHashMap.newKeySet();
        private final Set<Long> admitted = ConcurrentHashMap.newKeySet();
        private final List<Long> admissionOrder = new CopyOnWriteArrayList<>();
        private final Set<Long> decodeBlocked = ConcurrentHashMap.newKeySet();
        private volatile LongConsumer onSelection = ignored -> { };
        private final RequestScheduler scheduler;
        private final RoleType role;

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
            PrefillRoutingEntry a = endpoint("a", 0);
            PrefillRoutingEntry b = endpoint("b", 1);
            RequestRegistry lifecycle = mock(RequestRegistry.class);
            when(lifecycle.register(any())).thenAnswer(i -> {
                BalanceContext context = i.getArgument(0);
                CompletableFuture<Response> future = new CompletableFuture<>();
                requests.put(context.getRequestId(), future);
                return future;
            });
            when(lifecycle.claimAdmissionMutation(anyLong(), any())).thenAnswer(i -> mock(AdmissionMutation.class));
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
                ScheduledRequest item = mock(ScheduledRequest.class);
                ServerStatus status = new ServerStatus();
                status.setRole(role);
                when(item.prefill()).thenReturn(status);
                when(item.prefillEp()).thenReturn(a.endpoint());
                when(route.tryEnqueue(any(), any(), any())).thenAnswer(commit -> {
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
            groups.put(id, group);
            scheduler.submit(RequestLifecycleTestSupport.context(config, id));
        }

        private void release(String group) {
            availability.capacityChanged(PlacementKey.exact(role, group, group + ":8000"));
        }

        public void close() {
            requests.values().forEach(future -> future.complete(new Response()));
            scheduler.closePlacement();
        }

        private static PrefillRoutingEntry endpoint(String group, long slots) {
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            WorkerStatus status = mock(WorkerStatus.class);
            WorkerStatus.TopologySnapshot topology = mock(WorkerStatus.TopologySnapshot.class);
            when(status.topologySnapshot()).thenReturn(topology);
            when(topology.group()).thenReturn(group);
            when(endpoint.getStatus()).thenReturn(status);
            when(endpoint.availableRequestSlots()).thenReturn(slots);
            return new PrefillRoutingEntry(group + ":8000", endpoint);
        }
    }
}
