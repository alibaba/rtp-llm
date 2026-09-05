package org.flexlb.balance.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.ArgumentMatchers.any;

@Slf4j
class CostBasedDecodeStrategyTest {

    private ConfigService configService;
    private Map<String, WorkerStatus> decodeStatuses;

    @BeforeEach
    void setUp() {
        configService = new ConfigService();
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

    private CostBasedDecodeStrategy availableStrategy(EndpointRegistry registry) {
        return new CostBasedDecodeStrategy(new WorkerDirectory(registry));
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
        CostBasedDecodeStrategy costBasedDecodeStrategy = new CostBasedDecodeStrategy(
                engineWorkerStatus);

        BalanceContext balanceContext = context(1_000, 1_000L);

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

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
                new CostBasedDecodeStrategy(racing).select(
                        context(1_000, 1_001L), RoleType.DECODE, null);

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
        CostBasedDecodeStrategy costBasedDecodeStrategy = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(1_000, 1_000L);

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertNotNull(status.getServerIp());
    }

    @Test
    void should_handle_group_selection_when_group_parameter_provided() {
        WorkerStatus worker1 = createWorkerStatus("127.0.0.1", "group-a");

        decodeStatuses.put("127.0.0.1:8080", worker1);

        CostBasedDecodeStrategy costBasedDecodeStrategy = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(1_000, 1_000L);

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, "group-a");

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp());
    }

    @Test
    void should_use_exponential_decay_for_balanced_weight_distribution_when_cache_usage_differs() {
        registerWorker("127.0.0.1", 10_000, 9_500);
        registerWorker("127.0.0.2", 10_000, 8_500);
        CostBasedDecodeStrategy costBasedDecodeStrategy = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(1_000, 1_000L);

        int totalRuns = 10000;
        Map<String, Integer> selectionCount = new HashMap<>();

        for (int i = 0; i < totalRuns; i++) {
            balanceContext.getRequest().setRequestId(1000L + i);
            ServerStatus status = selectStatus(
                    costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

            if (status.isSuccess()) {
                String selectedIp = status.getServerIp();
                selectionCount.put(selectedIp, selectionCount.getOrDefault(selectedIp, 0) + 1);
            }
        }

        int worker1Count = selectionCount.getOrDefault("127.0.0.1", 0);
        int worker2Count = selectionCount.getOrDefault("127.0.0.2", 0);
        Assertions.assertTrue(worker1Count > worker2Count,
                "Worker with lower cache usage should be selected more frequently");

        double ratio = (double) worker1Count / worker2Count;
        Assertions.assertTrue(ratio >= 1.5 && ratio <= 3.0,
                "Weight ratio should be between 1.5-3.0, actual ratio: %.2f".formatted(ratio));
    }

    @Test
    void should_not_overflow_weight_when_decode_workers_have_large_kv_usage_gap() {
        int workerCount = 16;
        for (int i = 1; i <= workerCount; i++) {
            String ip = "127.0.0." + i;
            // With 15 workers using 800K tokens and one empty worker, the previous
            // average-centered formula evaluated exp(750), which overflows to Infinity.
            registerWorker(ip, 1_000_000, i == 1 ? 1_000_000 : 200_000);
        }
        EndpointRegistry registry = decodeRegistry();
        WorkerDirectory actual = new WorkerDirectory(registry);
        List<DecodeEndpoint.DecodeRoutingView> ordered = new ArrayList<>(
                actual.decodeRoutingSnapshot(null));
        DecodeEndpoint.DecodeRoutingView globalMinimum = ordered.stream()
                .filter(view -> view.address().equals("127.0.0.1:8080"))
                .findFirst()
                .orElseThrow();
        ordered.remove(globalMinimum);
        ordered.add(globalMinimum);
        Assertions.assertEquals(workerCount, ordered.size(),
                "the decode selector must retain the complete live fleet");
        WorkerDirectory fullFleet = Mockito.mock(WorkerDirectory.class);
        Mockito.when(fullFleet.decodeRoutingSnapshot(null)).thenReturn(ordered);
        Mockito.when(fullFleet.captureDecodeGeneration(any()))
                .thenAnswer(invocation -> actual.captureDecodeGeneration(
                        invocation.getArgument(
                                0, DecodeEndpoint.DecodeRoutingView.class)));
        CostBasedDecodeStrategy costBasedDecodeStrategy =
                new CostBasedDecodeStrategy(fullFleet);
        BalanceContext balanceContext = context(1, 10_000L);
        Request req = balanceContext.getRequest();

        for (int i = 0; i < 100; i++) {
            long requestId = 10_000L + i;
            req.setRequestId(requestId);
            ServerStatus status = Assertions.assertDoesNotThrow(
                    () -> selectStatus(
                            costBasedDecodeStrategy, balanceContext,
                            RoleType.DECODE, null));

            Assertions.assertTrue(status.isSuccess());
            Assertions.assertEquals("127.0.0.1", status.getServerIp(),
                    "The worker with the lowest KV usage should have the highest stable weight");
        }
    }

    @Test
    void should_skip_worker_with_insufficient_kv_cache_capacity() {
        configService.loadBalanceConfig().setScheduler(SchedulerConfig.direct());
        registerWorker("127.0.0.1", 1_000, 100);
        registerWorker("127.0.0.2", 1_000, 800);
        CostBasedDecodeStrategy costBasedDecodeStrategy = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(500, 2_000L);

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.2", status.getServerIp());
    }

    @Test
    void should_return_error_when_all_workers_kv_insufficient() {
        configService.loadBalanceConfig().setScheduler(SchedulerConfig.direct());
        registerWorker("127.0.0.1", 1_000, 50);
        registerWorker("127.0.0.2", 1_000, 100);
        CostBasedDecodeStrategy costBasedDecodeStrategy = availableStrategy(decodeRegistry());
        BalanceContext balanceContext = context(200, 3_000L);

        ServerStatus status = selectStatus(
                costBasedDecodeStrategy, balanceContext, RoleType.DECODE, null);

        Assertions.assertNull(status);
    }

    @Test
    void queueRejectsSequenceBeyondEveryKnownPhysicalCapacity() {
        registerWorker("127.0.0.1", 128L, 128L);
        registerWorker("127.0.0.2", 256L, 256L);
        CostBasedDecodeStrategy strategy = availableStrategy(decodeRegistry());
        BalanceContext context = context(257L, 3_050L);

        PlacementResult<SelectedRole, RoleType> result = strategy.select(
                context, RoleType.DECODE, null);

        Assertions.assertEquals(PlacementResult.Status.REJECTED, result.status());
        Assertions.assertTrue(result.rejection().getErrorMessage().contains("257"));
        Assertions.assertTrue(result.rejection().getErrorMessage().contains("256"));
    }

    @Test
    void unknownPhysicalCapacityDoesNotBecomeATerminalRejection() {
        registerWorker("127.0.0.1", 0L, 0L);
        CostBasedDecodeStrategy strategy = availableStrategy(decodeRegistry());

        PlacementResult<SelectedRole, RoleType> result = strategy.select(
                context(1_000_000L, 3_051L), RoleType.DECODE, null);

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

        Assertions.assertFalse(CostBasedDecodeStrategy.hasDecodeCapacity(
                configService.loadBalanceConfig(), endpoint.routingView(), false));

        CostBasedDecodeStrategy strategy = new CostBasedDecodeStrategy(
                new WorkerDirectory(registry));

        BalanceContext context = context(100, 3L);
        Request request = context.getRequest();

        PlacementResult<SelectedRole, RoleType> fifoPlacement = strategy.select(
                context, RoleType.DECODE, null);
        Assertions.assertEquals(
                PlacementResult.Status.SUCCESS, fifoPlacement.status());
        SelectedRole fifoSelection = fifoPlacement.value();
        ServerStatus fifoResult = fifoSelection.serverStatus();
        Assertions.assertTrue(fifoResult.isSuccess());
        Assertions.assertEquals(1_000L, fifoSelection.decodeTotalKv());
        Assertions.assertFalse(endpoint.layeredAdmissionView().isQueued(3L),
                "selection must not mutate Decode reservation ownership");
        fifoSelection.close();

        QueueOrderingConfig preemptiveOrdering =
                QueueOrderingConfig.priority();
        preemptiveOrdering.setPreemption(new PreemptionConfig());
        configService.loadBalanceConfig().queueScheduler()
                .setOrdering(preemptiveOrdering);
        request.setRequestId(4L);
        PlacementResult<SelectedRole, RoleType> priorityPlacement =
                strategy.select(context, RoleType.DECODE, null);
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
        CostBasedDecodeStrategy strategy = availableStrategy(registry);
        BalanceContext context = context(300L, 5L);

        ServerStatus selected = selectStatus(
                strategy, context, RoleType.DECODE, null);

        Assertions.assertNotNull(selected);
        Assertions.assertEquals("127.0.0.2", selected.getServerIp(),
                "routing must project this request through the exact Decode gate");
    }

    @Test
    void softQueuePlacementBalancesTwentyThousandProjectedOwners() {
        registerWorker("127.0.0.1", 1_000, 1_000);
        registerWorker("127.0.0.2", 1_000, 1_000);
        EndpointRegistry registry = decodeRegistry();
        DecodeEndpoint hotspot = decodeEndpoint(
                registry, "127.0.0.1:8080");
        for (long requestId = 1L; requestId <= 64L; requestId++) {
            reserveQueued(hotspot, requestId, 0L, 0L, 50);
        }
        Assertions.assertEquals(64, hotspot.routingView().totalLoad());
        Assertions.assertEquals(
                0, decodeEndpoint(registry, "127.0.0.2:8080")
                        .routingView().totalLoad());

        CostBasedDecodeStrategy strategy = availableStrategy(registry);
        BalanceContext context = context(0L, 10_000L);
        Request request = context.getRequest();

        DecodeEndpoint peer = decodeEndpoint(
                registry, "127.0.0.2:8080");
        for (int index = 0; index < 20_000; index++) {
            long requestId = 10_000L + index;
            request.setRequestId(requestId);
            ServerStatus selected = selectStatus(
                    strategy, context, RoleType.DECODE, null);
            Assertions.assertNotNull(selected);
            DecodeEndpoint selectedEndpoint = decodeEndpoint(registry,
                    selected.getServerIp() + ":8080");
            reserveQueued(selectedEndpoint, requestId, 0L, 0L, 50);
        }

        int finalSkew = Math.abs(
                hotspot.routingView().totalLoad()
                        - peer.routingView().totalLoad());
        Assertions.assertTrue(finalSkew <= 8,
                "soft load weighting should converge without a hard minimum tier; skew="
                        + finalSkew);
    }

    @Test
    void oneLowerSnapshotDoesNotHerdEveryConcurrentPlan() {
        registerWorker("127.0.0.1", 1_000, 1_000);
        registerWorker("127.0.0.2", 1_000, 1_000);
        EndpointRegistry registry = decodeRegistry();
        reserveQueued(
                decodeEndpoint(registry, "127.0.0.2:8080"),
                91L, 0L, 0L, 50);
        CostBasedDecodeStrategy strategy = availableStrategy(registry);
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

        Assertions.assertTrue(lowerLoadSelections > higherLoadSelections,
                "lower load should retain a soft preference");
        Assertions.assertTrue(higherLoadSelections > 0,
                "a shared snapshot must not collapse every plan onto one endpoint");
    }

    @Test
    void singleCandidateSkipsOutlierRejection() {
        // Upstream's directory refactor replaced the shared static ledger
        // (EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS) with the per-test
        // decodeStatuses map, so this outlier-rejection guard rides the same
        // local-directory contract as every other test here.
        WorkerStatus worker = createWorkerStatus("127.0.0.1");
        setKv(worker, 10_000, 10_000);
        decodeStatuses.put("127.0.0.1:8080", worker);

        EndpointRegistry registry = createDecodeRegistry(decodeStatuses);
        DecodeEndpoint endpoint = registry.getDecode("127.0.0.1:8080");
        // n == 1 with the upstream self-inclusive average: the average IS
        // the engine's own load, so own > multiplier * avg can never hold —
        // a lone engine always stays selectable regardless of its load.
        for (int i = 0; i < 6; i++) {
            reservePinned(endpoint, 400L + i, 0, 0, 50);
        }

        // Upstream dropped the ResourceMeasureFactory indirection: the
        // strategy now takes the DecodeResourceMeasure directly, so the
        // mock measure is wired straight into the constructor.
        DecodeResourceMeasure measure = Mockito.mock(DecodeResourceMeasure.class);
        allowDecodeSelection(measure);
        CostBasedDecodeStrategy strategy = new CostBasedDecodeStrategy(
                new WorkerDirectory(registry), measure);

        Request request = new Request();
        request.setSeqLen(1);
        request.setRequestId(500L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(configService.loadBalanceConfig());

        ServerStatus status = selectStatus(
                strategy, context, RoleType.DECODE, null);
        Assertions.assertTrue(status.isSuccess());
        Assertions.assertEquals("127.0.0.1", status.getServerIp(),
                "a lone engine's self-inclusive average equals its own load, so it must stay selectable");
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
            endpoint.tryReserveQueuedPinned(
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
            CostBasedDecodeStrategy strategy,
            BalanceContext context,
            RoleType role,
            String group) {
        PlacementResult<SelectedRole, RoleType> result =
                strategy.select(context, role, group);
        if (result.status() != PlacementResult.Status.SUCCESS) {
            return null;
        }
        try (SelectedRole selected = result.value()) {
            return selected.serverStatus();
        }
    }
}
