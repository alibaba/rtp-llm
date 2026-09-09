package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.Mockito;

import java.util.EnumSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Slf4j
class DecodeSelectorTest {

    private ConfigService configService;
    private Map<String, WorkerStatus> decodeStatuses;

    @BeforeEach
    void setUp() {
        configService = Mockito.mock(ConfigService.class);
        Mockito.when(configService.loadBalanceConfig()).thenReturn(StrategyTestSupport.config());
        decodeStatuses = new HashMap<>();
    }

    WorkerStatus createWorkerStatus(String ip) {
        return createWorkerStatus(ip, null);
    }

    WorkerStatus createWorkerStatus(String ip, String group) {
        return StrategyTestSupport.workerStatus(
                RoleType.DECODE, group, ip, 8080, 9090,
                true, 0L, 0L);
    }

    /** Create an EndpointRegistry with DecodeEndpoints registered for each WorkerStatus entry. */
    private EndpointRegistry createDecodeRegistry(Map<String, WorkerStatus> workerMap) {
        EndpointRegistry registry = StrategyTestSupport.endpointRegistry(configService);
        for (Map.Entry<String, WorkerStatus> entry : workerMap.entrySet()) {
            WorkerStatus ws = entry.getValue();
            StrategyTestSupport.publishEndpoint(registry,
                    RoleType.DECODE, entry.getKey(), ws);
        }
        return registry;
    }

    private WorkerStatus registerWorker(String ip, long totalKv, long availableKv) {
        WorkerStatus worker = createWorkerStatus(ip);
        setKv(worker, totalKv, availableKv);
        decodeStatuses.put(ip + ":8080", worker);
        return worker;
    }

    private EndpointRegistry decodeRegistry() {
        return createDecodeRegistry(decodeStatuses);
    }

    private DecodeSelector availableStrategy(EndpointRegistry registry) {
        return new DecodeSelector(new WorkerDirectory(registry));
    }

    private BalanceContext context(long sequenceLength, long requestId) {
        Request request = new Request();
        request.setSeqLen(sequenceLength);
        request.setRequestId(requestId);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(configService.loadBalanceConfig());
        return context;
    }

    @Test
    void should_handle_empty_worker_map_when_no_workers_available() {
        EndpointRegistry emptyRegistry = StrategyTestSupport.endpointRegistry(configService);
        WorkerDirectory engineWorkerStatus = new WorkerDirectory(emptyRegistry);
        DecodeSelector decodeSelector = new DecodeSelector(
                engineWorkerStatus);

        BalanceContext balanceContext = context(1_000, 1_000L);

        ServerStatus status = selectStatus(
                decodeSelector, balanceContext, RoleType.DECODE, null);

        Assertions.assertNull(status);
    }

    @Test
    void retriesOneCaptureConflictBeforePublishingTheSelection() {
        registerWorker("127.0.0.1", 10_000, 9_000);
        registerWorker("127.0.0.2", 10_000, 9_000);
        EndpointRegistry registry = decodeRegistry();
        WorkerDirectory actual = new WorkerDirectory(registry);
        Map<String, DecodeEndpoint.DecodeRoutingView> views = new HashMap<>();
        for (DecodeEndpoint.DecodeRoutingView view
                : actual.decodeRoutingSnapshot(null)) {
            views.put(view.address(), view);
        }
        DecodeEndpoint.DecodeRoutingView stale = views.get("127.0.0.1:8080");
        DecodeEndpoint.DecodeRoutingView replacement =
                views.get("127.0.0.2:8080");
        WorkerEndpoint.GenerationPin replacementPin =
                Mockito.mock(WorkerEndpoint.GenerationPin.class);
        Mockito.when(replacementPin.endpoint()).thenReturn(
                decodeEndpoint(registry, replacement.address()));
        Mockito.when(replacementPin.generationId()).thenReturn(
                replacement.generationId());
        WorkerDirectory racing = Mockito.mock(WorkerDirectory.class);
        Mockito.when(racing.decodeRoutingSnapshot(null))
                .thenReturn(List.of(stale))
                .thenReturn(List.of(replacement));
        Mockito.when(racing.captureDecodeGeneration(stale)).thenReturn(null);
        Mockito.when(racing.captureDecodeGeneration(replacement))
                .thenReturn(replacementPin);

        PlacementResult<SelectedRole, RoleType> result =
                new DecodeSelector(racing).select(
                        DecodeBinding.capture(context(1_000, 1_001L)), null);

        Assertions.assertEquals(PlacementResult.Status.SUCCESS, result.status());
        SelectedRole selected = result.value();
        Assertions.assertEquals(
                "127.0.0.2", selected.serverStatus().getServerIp());
        Mockito.verify(replacementPin, Mockito.never()).close();
        selected.close();
        Mockito.verify(replacementPin).close();
        Mockito.verify(racing, Mockito.times(2)).decodeRoutingSnapshot(null);
        Mockito.verify(racing).captureDecodeGeneration(stale);
        Mockito.verify(racing).captureDecodeGeneration(replacement);
    }

    @Test
    void should_use_uniform_distribution_when_all_cache_usages_are_equal() {
        registerWorker("127.0.0.1", 10_000, 9_000);
        registerWorker("127.0.0.2", 10_000, 9_000);
        registerWorker("127.0.0.3", 10_000, 9_000);
        DecodeSelector decodeSelector = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(1_000, 1_000L);

        ServerStatus status = selectStatus(
                decodeSelector, balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_handle_group_selection_when_group_parameter_provided() {
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1", "group-a");

        decodeStatuses.put("127.0.0.1:8080", worker1);

        DecodeSelector decodeSelector = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(1_000, 1_000L);

        ServerStatus status = selectStatus(
                decodeSelector, balanceContext, RoleType.DECODE, "group-a");

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }

    @Test
    void roundRobinDoesNotWeightAdmissibleWorkersByKvUsage() {
        registerWorker("127.0.0.1", 10_000, 9_500);
        registerWorker("127.0.0.2", 10_000, 8_500);
        DecodeSelector strategy = availableStrategy(decodeRegistry());
        Map<String, Integer> counts = new HashMap<>();
        for (int i = 0; i < 100; i++) {
            ServerStatus selected = selectStatus(strategy, context(1000L, 1000L + i),
                    RoleType.DECODE, null);
            counts.merge(selected.getServerIp(), 1, Integer::sum);
        }
        Assertions.assertEquals(Map.of("127.0.0.1", 50, "127.0.0.2", 50), counts);
    }

    @Test
    void roundRobinRetainsTheCompleteAdmissibleFleet() {
        for (int i = 1; i <= 16; i++) {
            registerWorker("127.0.0." + i, 1_000_000, i == 1 ? 1_000_000 : 200_000);
        }
        DecodeSelector strategy = availableStrategy(decodeRegistry());
        java.util.Set<String> selected = new java.util.HashSet<>();
        for (int i = 0; i < 16; i++) {
            selected.add(selectStatus(strategy, context(1L, 10_000L + i),
                    RoleType.DECODE, null).getServerIp());
        }
        Assertions.assertEquals(16, selected.size());
    }

    @Test
    void should_skip_worker_with_insufficient_kv_cache_capacity() {
        configService.loadBalanceConfig().setScheduler(SchedulerConfig.direct());
        registerWorker("127.0.0.1", 1_000, 100);
        registerWorker("127.0.0.2", 1_000, 800);
        DecodeSelector decodeSelector = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(500, 2_000L);

        ServerStatus status = selectStatus(
                decodeSelector, balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.2", status.getServerIp());
    }

    @Test
    void should_return_error_when_all_workers_kv_insufficient() {
        configService.loadBalanceConfig().setScheduler(SchedulerConfig.direct());
        registerWorker("127.0.0.1", 1_000, 50);
        registerWorker("127.0.0.2", 1_000, 100);
        DecodeSelector decodeSelector = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(200, 3_000L);

        ServerStatus status = selectStatus(
                decodeSelector, balanceContext, RoleType.DECODE, null);

        Assertions.assertNull(status);
    }

    @Test
    void queueRejectsSequenceBeyondEveryKnownPhysicalCapacity() {
        registerWorker("127.0.0.1", 128L, 128L);
        registerWorker("127.0.0.2", 256L, 256L);
        DecodeSelector strategy = availableStrategy(decodeRegistry());
        BalanceContext context = context(257L, 3_050L);
        context.getRequest().setMaxNewTokens(1);

        PlacementResult<SelectedRole, RoleType> result = strategy.select(
                DecodeBinding.capture(context), null);

        Assertions.assertEquals(PlacementResult.Status.REJECTED, result.status());
        Assertions.assertTrue(result.rejection().getErrorMessage().contains("demand=258"));
        Assertions.assertTrue(result.rejection().getErrorMessage().contains("maximum=230"));
    }

    @Test
    void unknownPhysicalCapacityDoesNotBecomeATerminalRejection() {
        registerWorker("127.0.0.1", 0L, 0L);
        DecodeSelector strategy = availableStrategy(decodeRegistry());

        PlacementResult<SelectedRole, RoleType> result = strategy.select(
                DecodeBinding.capture(context(1_000_000L, 3_051L)), null);

        Assertions.assertEquals(PlacementResult.Status.SUCCESS, result.status());
        result.value().close();
    }

    @Test
    void queuePlanningSelectsExactEndpointWithoutTakingCapacity() {
        registerWorker("127.0.0.1", 1_000, 1_000);
        EndpointRegistry registry = decodeRegistry();
        DecodeEndpoint endpoint = decodeEndpoint(
                registry, "127.0.0.1:8080");
        reserveQueued(endpoint, 1L, 400, 700, 50);
        reserveQueued(endpoint, 2L, 400, 700, 50);

        Assertions.assertTrue(endpoint.routingView().realKvUsed()
                > new DecodeEndpoint.AdmissionCapacity(0L, 90L).kvBudget(endpoint.realKvTotal()));
        Assertions.assertEquals(0L, endpoint.routingView().engineFacingKvUsed());

        DecodeSelector strategy = new DecodeSelector(
                new WorkerDirectory(registry));

        BalanceContext context = context(100, 3L);
        Request request = context.getRequest();

        PlacementResult<SelectedRole, RoleType> fifoPlacement = strategy.select(
                DecodeBinding.capture(context), null);
        Assertions.assertEquals(
                PlacementResult.Status.SUCCESS, fifoPlacement.status());
        SelectedRole fifoSelection = fifoPlacement.value();
        ServerStatus fifoResult = fifoSelection.serverStatus();
        Assertions.assertTrue(fifoResult.isSuccess());
        Assertions.assertEquals(request.getRequestId(), fifoResult.getRequestId());
        Assertions.assertFalse(endpoint.layeredAdmissionView().isQueued(3L),
                "selection must not mutate Decode reservation ownership");
        fifoSelection.close();

        QueueOrderingConfig preemptiveOrdering =
                QueueOrderingConfig.priority();
        preemptiveOrdering.setPreemption(preemption());
        configService.loadBalanceConfig().queueScheduler()
                .setOrdering(preemptiveOrdering);
        request.setRequestId(4L);
        PlacementResult<SelectedRole, RoleType> priorityPlacement =
                strategy.select(DecodeBinding.capture(context), null);
        Assertions.assertEquals(
                PlacementResult.Status.SUCCESS, priorityPlacement.status());
        Assertions.assertFalse(endpoint.layeredAdmissionView().isQueued(4L),
                "priority planning must leave capacity acquisition to commit");
        priorityPlacement.value().close();
    }

    @Test
    void nonPreemptiveQueueProjectsCurrentRequestBeforeOwnershipTier() {
        registerWorker("127.0.0.1", 1_000, 200);
        registerWorker("127.0.0.2", 1_000, 1_000);
        EndpointRegistry registry = decodeRegistry();
        // Bias the old least-ownership tier toward the endpoint which cannot
        // fit this request. Request-aware capacity must win that disagreement.
        reserveQueued(
                decodeEndpoint(registry, "127.0.0.2:8080"),
                91L, 0L, 0L, 50);
        DecodeSelector strategy = availableStrategy(registry);
        BalanceContext context = context(300L, 5L);

        ServerStatus selected = selectStatus(
                strategy, context, RoleType.DECODE, null);

        Assertions.assertNotNull(selected);
        Assertions.assertEquals("127.0.0.2", selected.getServerIp(),
                "routing must project this request through the exact Decode gate");
    }

    @ParameterizedTest
    @EnumSource(QueuePolicy.class)
    void queueSelectsFeasibleEndpointsBeforeWaitingForTransientCapacity(QueuePolicy policy) {
        configureQueue(policy);
        registerWorker("127.0.0.1", 1000, 100);
        registerWorker("127.0.0.2", 1000, 100);
        EndpointRegistry registry = decodeRegistry();
        DecodeSelector strategy = availableStrategy(registry);
        Map<String, Integer> counts = new HashMap<>();
        for (int index = 0; index < 10; index++) {
            long requestId = 123L + index;
            PlacementResult<SelectedRole, RoleType> result = strategy.select(
                    DecodeBinding.capture(context(100L, requestId)), null);
            Assertions.assertEquals(PlacementResult.Status.SUCCESS, result.status());
            try (SelectedRole selected = result.value()) {
                counts.merge(selected.serverStatus().getServerIp(), 1, Integer::sum);
                Assertions.assertFalse(decodeEndpoint(registry,
                        selected.serverStatus().getServerIp() + ":8080")
                        .layeredAdmissionView().isQueued(requestId), "selection cannot claim capacity");
            }
        }
        Assertions.assertEquals(Map.of("127.0.0.1", 5, "127.0.0.2", 5), counts);
    }

    @ParameterizedTest
    @EnumSource(QueuePolicy.class)
    void waitingDoesNotMakePhysicallyImpossibleEndpointsEligible(QueuePolicy policy) {
        configureQueue(policy);
        registerWorker("127.0.0.1", 100L, 100L);
        registerWorker("127.0.0.2", 1000L, 0L);
        DecodeSelector strategy = availableStrategy(decodeRegistry());
        for (long requestId = 200L; requestId < 204L; requestId++) {
            ServerStatus selected = selectStatus(strategy, context(200L, requestId), RoleType.DECODE, null);
            Assertions.assertNotNull(selected);
            Assertions.assertEquals("127.0.0.2", selected.getServerIp());
        }
    }

    private enum QueuePolicy { FIFO, PRIORITY, PREEMPTIVE }

    private void configureQueue(QueuePolicy policy) {
        QueueOrderingConfig ordering = policy == QueuePolicy.FIFO
                ? new QueueOrderingConfig() : QueueOrderingConfig.priority();
        if (policy == QueuePolicy.PRIORITY) { ordering.setPreemption(null); }
        if (policy == QueuePolicy.PREEMPTIVE) { ordering.setPreemption(preemption()); }
        configService.loadBalanceConfig().queueScheduler().setOrdering(ordering);
    }

    private static PreemptionConfig preemption() {
        PreemptionConfig config = new PreemptionConfig();
        config.setAllowedVictimStages(EnumSet.of(VictimStage.DECODE_RESERVED));
        return config;
    }

    @Test
    void oneLowerSnapshotDoesNotHerdEveryConcurrentPlan() {
        registerWorker("127.0.0.1", 1_000, 1_000);
        registerWorker("127.0.0.2", 1_000, 1_000);
        EndpointRegistry registry = decodeRegistry();
        reserveQueued(
                decodeEndpoint(registry, "127.0.0.2:8080"),
                91L, 0L, 0L, 50);
        DecodeSelector strategy = availableStrategy(registry);
        BalanceContext context = context(0L, 30_000L);
        int lowerLoadSelections = 0;
        int higherLoadSelections = 0;

        for (int index = 0; index < 1_000; index++) {
            context.getRequest().setRequestId(30_000L + index);
            ServerStatus selected = selectStatus(
                    strategy, context, RoleType.DECODE, null);
            if ("127.0.0.1".equals(selected.getServerIp())) {
                lowerLoadSelections++;
            } else {
                higherLoadSelections++;
            }
        }

        Assertions.assertEquals(500, lowerLoadSelections);
        Assertions.assertEquals(500, higherLoadSelections);
    }

    @Test
    void highLoadRemainsAdmissibleBelowTheConfiguredCapacity() {
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        setKv(worker, 10_000, 10_000);
        decodeStatuses.put("127.0.0.1:8080", worker);

        EndpointRegistry registry = createDecodeRegistry(decodeStatuses);
        DecodeEndpoint endpoint = decodeEndpoint(registry, "127.0.0.1:8080");
        for (int i = 0; i < 6; i++) {
            reservePinned(endpoint, 400L + i, 0, 0, 50);
        }

        DecodeSelector strategy = availableStrategy(registry);

        Request request = new Request();
        request.setSeqLen(1);
        request.setRequestId(500L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(configService.loadBalanceConfig());

        ServerStatus status = selectStatus(
                strategy, context, RoleType.DECODE, null);
        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }

    @Test
    void fullOutputDemandCannotWaitForeverBehindAnImpossibleKvBudget() {
        registerWorker("127.0.0.1", 10_000L, 10_000L);
        BalanceContext context = context(500L, 991L);
        context.getRequest().setMaxNewTokens(10_000);
        Assertions.assertEquals(PlacementResult.Status.REJECTED,
                availableStrategy(decodeRegistry()).select(DecodeBinding.capture(context), null).status());
    }

    @Test
    void zeroKvBudgetRejectsPositiveDemand() {
        configService.loadBalanceConfig().getRouter().getRoles().getDecode()
                .getAvailability().setMaxKvUsagePercent(0);
        registerWorker("127.0.0.1", 10_000L, 10_000L);
        Assertions.assertEquals(PlacementResult.Status.REJECTED,
                availableStrategy(decodeRegistry()).select(DecodeBinding.capture(context(1L, 992L)), null).status());
    }

    private static void setKv(
            WorkerStatus worker, long totalKv, long availableKv) {
        StrategyTestSupport.publish(worker, StrategyTestSupport.response(
                RoleType.DECODE, true, availableKv, totalKv,
                Math.max(1L, worker.appliedStatusCursor().statusVersion() + 1L)));
    }

    private static void reserveQueued(
            DecodeEndpoint endpoint,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        try (var pin = endpoint.tryPinGeneration()) {
            endpoint.tryReservePlacementPinned(
                    pin, requestId, kvTokens, expectedKvTokens, priority);
        }
    }

    private static DecodeEndpoint decodeEndpoint(
            EndpointRegistry registry,
            String address) {
        return (DecodeEndpoint) registry.get(RoleType.DECODE, address);
    }

    /** Non-queued inflight reservation: each call raises engineLoad by one. */
    private static void reservePinned(
            DecodeEndpoint endpoint,
            long requestId,
            long kvTokens,
            long expectedKvTokens,
            int priority) {
        try (var pin = endpoint.tryPinGeneration()) {
            endpoint.reservePinned(
                    pin, requestId, kvTokens, expectedKvTokens, priority);
        }
    }

    private static ServerStatus selectStatus(
            DecodeSelector strategy,
            BalanceContext context,
            RoleType role,
            String group) {
        PlacementResult<SelectedRole, RoleType> result =
                strategy.select(DecodeBinding.capture(context), group);
        if (result.status() != PlacementResult.Status.SUCCESS) {
            return null;
        }
        try (SelectedRole selected = result.value()) {
            return selected.serverStatus();
        }
    }
}
