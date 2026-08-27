package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.cache.domain.CacheMatch;
import org.flexlb.cache.domain.EngineGeneration;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/**
 * Queue ownership contract for transient worker dispatch pressure.
 *
 * <p>These tests intentionally exercise the production routing measures and
 * scheduler. A temporary worker capacity limit is backpressure, not a
 * terminal routing failure: the original request must remain globally owned
 * until capacity is released, cancelled, expired, or rejected by the global
 * queue bound.
 */
class TransientCapacityQueueContractTest {

    @Test
    @Timeout(30)
    void prefillBindingLimitWaitsAndRunsOriginalRequestAfterCapacityReturns()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            assertEquals(1L,
                    fixture.prefill.admissionPendingRequestCount(),
                    "fixture must begin at the Prefill binding limit");

            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(fixture.context(101L));

            assertFalse(waiting.isDone(), () ->
                    "temporary Prefill binding pressure must keep the "
                            + "original request in the global queue; observed "
                            + "terminal code=" + waiting.join().getCode());
            assertEquals(0, fixture.submission.commands().size(),
                    "a capacity-blocked request must not cross the dispatch boundary");

            fixture.releaseExternalCapacity(RoleType.PREFILL);

            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(),
                    "the original request must run after Prefill capacity returns");
            assertEquals(List.of(101L), fixture.submission.requestIds(),
                    "capacity recovery must dispatch the original request exactly once");
        }
    }

    @Test
    @Timeout(30)
    void decodeDispatchLimitWaitsAndRunsOriginalRequestAfterCapacityReturns()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            assertEquals(1L, fixture.decode.routingView().engineLoad(),
                    "fixture must begin at the Decode dispatch limit");

            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(fixture.context(201L));

            assertFalse(waiting.isDone(), () ->
                    "temporary Decode dispatch pressure must keep the "
                            + "original request in the global queue; observed "
                            + "terminal code=" + waiting.join().getCode());
            assertEquals(0, fixture.submission.commands().size(),
                    "a capacity-blocked request must not cross the dispatch boundary");

            fixture.releaseExternalCapacity(RoleType.DECODE);

            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess(),
                    "the original request must run after Decode capacity returns");
            assertEquals(List.of(201L), fixture.submission.requestIds(),
                    "capacity recovery must dispatch the original request exactly once");
        }
    }

    @Test
    @Timeout(30)
    void releasedPrefillSlotGoesToOlderWaiterAtTheSamePriority()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            fixture.submission.holdCompletions();

            CompletableFuture<Response> older =
                    fixture.runtime.scheduler().submit(
                            fixture.context(251L, 50));
            CompletableFuture<Response> later =
                    fixture.runtime.scheduler().submit(
                            fixture.context(252L, 50));

            assertFalse(older.isDone(),
                    "the older request must remain placement-waiting");
            assertFalse(later.isDone(),
                    "the later request must remain placement-waiting");
            assertEquals(2,
                    fixture.runtime.scheduler().getQueuedRequestCount(),
                    "both requests must be scheduler-owned before release");
            assertEquals(0, fixture.submission.commands().size(),
                    "neither waiter may dispatch while Prefill is full");

            fixture.releaseExternalCapacity(RoleType.PREFILL);

            assertTrue(fixture.submission.awaitCommands(
                            1, 2, TimeUnit.SECONDS),
                    "one released Prefill slot must wake one waiter");
            assertEquals(List.of(251L), fixture.submission.requestIds(),
                    "same-priority placement must preserve arrival FIFO");
            assertFalse(fixture.submission.awaitCommands(
                            1, 250, TimeUnit.MILLISECONDS),
                    "the held first dispatch must keep the single slot full");
            assertEquals(1, fixture.submission.commands().size(),
                    "one physical Prefill slot must have one winner");
        }
    }

    @Test
    @Timeout(30)
    void releasedPrefillSlotGoesToLaterHigherPriorityWaiter()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            assertTrue(fixture.config.isPriorityOrdering(),
                    "this contract requires PRIORITY queue ordering");
            fixture.submission.holdCompletions();

            CompletableFuture<Response> older =
                    fixture.runtime.scheduler().submit(
                            fixture.context(261L, 50));
            CompletableFuture<Response> laterHigherPriority =
                    fixture.runtime.scheduler().submit(
                            fixture.context(262L, 80));

            assertFalse(older.isDone(),
                    "the older request must remain placement-waiting");
            assertFalse(laterHigherPriority.isDone(),
                    "the higher-priority request must remain placement-waiting");
            assertEquals(2,
                    fixture.runtime.scheduler().getQueuedRequestCount(),
                    "both requests must be scheduler-owned before release");
            assertEquals(0, fixture.submission.commands().size(),
                    "neither waiter may dispatch while Prefill is full");

            fixture.releaseExternalCapacity(RoleType.PREFILL);

            assertTrue(fixture.submission.awaitCommands(
                            1, 2, TimeUnit.SECONDS),
                    "one released Prefill slot must wake one waiter");
            assertEquals(List.of(262L), fixture.submission.requestIds(),
                    "an undecided higher-priority waiter must precede an "
                            + "older lower-priority waiter");
            assertFalse(fixture.submission.awaitCommands(
                            1, 250, TimeUnit.MILLISECONDS),
                    "the held first dispatch must keep the single slot full");
            assertEquals(1, fixture.submission.commands().size(),
                    "one physical Prefill slot must have one winner");
        }
    }

    @Test
    @Timeout(30)
    void releasedDecodeSlotGoesToOlderWaiterAtTheSamePriority()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            fixture.submission.holdCompletions();

            CompletableFuture<Response> older =
                    fixture.runtime.scheduler().submit(
                            fixture.context(271L, 50));
            CompletableFuture<Response> later =
                    fixture.runtime.scheduler().submit(
                            fixture.context(272L, 50));

            assertFalse(older.isDone(),
                    "the older request must remain placement-waiting");
            assertFalse(later.isDone(),
                    "the later request must remain placement-waiting");
            assertEquals(2,
                    fixture.runtime.scheduler().getQueuedRequestCount(),
                    "both requests must be scheduler-owned before release");
            assertEquals(0, fixture.submission.commands().size(),
                    "neither waiter may dispatch while Decode is full");

            fixture.releaseExternalCapacity(RoleType.DECODE);

            assertTrue(fixture.submission.awaitCommands(
                            1, 2, TimeUnit.SECONDS),
                    "one released Decode slot must wake one waiter");
            assertEquals(List.of(271L), fixture.submission.requestIds(),
                    "same-priority Decode placement must preserve arrival FIFO");
            assertFalse(fixture.submission.awaitCommands(
                            1, 250, TimeUnit.MILLISECONDS),
                    "the held first dispatch must keep the single slot full");
            assertEquals(1, fixture.submission.commands().size(),
                    "one physical Decode slot must have one winner");
        }
    }

    @Test
    @Timeout(30)
    void cacheAffinityWaitsForThePreferredPrefillInsteadOfSpillingCold()
            throws Exception {
        try (CacheAffinityFixture fixture = new CacheAffinityFixture()) {
            assertEquals(1L, fixture.hot.admissionPendingRequestCount(),
                    "cache leader must begin at its binding limit");
            assertEquals(0L, fixture.cold.admissionPendingRequestCount(),
                    "cold fallback must remain immediately bindable");

            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(fixture.context(301L));

            assertFalse(waiting.isDone(), () ->
                    "a cache-affine request must wait for its preferred Prefill"
                            + " instead of spilling cold; observed Prefill="
                            + prefillIp(waiting.join()));
            assertEquals(0, fixture.submission.commands().size(),
                    "affinity hold must not cross the dispatch boundary");

            fixture.releaseHotCapacity();

            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals(CacheAffinityFixture.HOT_IP, prefillIp(response),
                    "capacity recovery must preserve the original cache leader");
            assertEquals(List.of(301L), fixture.submission.requestIds(),
                    "affinity recovery must dispatch the original request exactly once");
        }
    }

    @Test
    @Timeout(30)
    void releasedHotSlotCannotBeStolenByAConcurrentColdEligibleArrival()
            throws Exception {
        try (CacheAffinityFixture fixture = new CacheAffinityFixture()) {
            fixture.submission.holdCompletions();
            CompletableFuture<Response> hotWaiter =
                    fixture.runtime.scheduler().submit(fixture.context(311L));

            assertFalse(hotWaiter.isDone(),
                    "the cache-hot request must remain scheduler-owned");
            assertEquals(0, fixture.submission.commands().size(),
                    "the hot request must not spill to the cold worker");

            CyclicBarrier start = new CyclicBarrier(2);
            try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
                Future<?> release = executor.submit(() -> {
                    await(start);
                    fixture.releaseHotCapacity();
                });
                Future<CompletableFuture<Response>> coldArrival =
                        executor.submit(() -> {
                            await(start);
                            return fixture.runtime.scheduler().submit(
                                    fixture.coldContext(312L));
                        });

                release.get(2, TimeUnit.SECONDS);
                CompletableFuture<Response> coldFuture =
                        coldArrival.get(2, TimeUnit.SECONDS);
                assertFalse(coldFuture.isDone(),
                        "held delivery keeps the cold request active");
            }

            assertTrue(fixture.submission.awaitCommands(
                            2, 2, TimeUnit.SECONDS),
                    "both independent Prefill slots must be dispatched");
            assertEquals(CacheAffinityFixture.HOT_IP,
                    fixture.submission.prefillIp(311L),
                    "the released hot slot belongs to the older affine waiter");
            assertEquals(CacheAffinityFixture.COLD_IP,
                    fixture.submission.prefillIp(312L),
                    "the concurrent cold-affine arrival must use its own leader");
            assertEquals(2, fixture.submission.commands().size(),
                    "each request must cross the dispatch boundary exactly once");
        }
    }

    @Test
    @Timeout(30)
    void duplicateCapacitySignalsGiveOneReleasedSlotToAtMostOneWaiter()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            fixture.submission.holdCompletions();
            List<CompletableFuture<Response>> waiters = IntStream.range(0, 16)
                    .mapToObj(index -> fixture.runtime.scheduler().submit(
                            fixture.context(400L + index, 50)))
                    .toList();

            assertTrue(waiters.stream().noneMatch(CompletableFuture::isDone),
                    "temporary Prefill pressure must retain all 16 waiters");
            assertEquals(0, fixture.submission.commands().size());

            fixture.releaseExternalCapacityRepeatedly(
                    RoleType.PREFILL, 16);

            assertTrue(fixture.submission.awaitCommands(
                            1, 2, TimeUnit.SECONDS),
                    "one physical release must wake one waiter");
            assertFalse(fixture.submission.awaitCommands(
                            1, 250, TimeUnit.MILLISECONDS),
                    "duplicate status signals must not create another slot");
            assertEquals(1, fixture.submission.commands().size(),
                    "one released Prefill slot must have at most one winner");
            assertEquals(1, fixture.submission.requestIds().stream()
                            .distinct().count(),
                    "the exact request generation must be dispatched once");
        }
    }

    private static final class Fixture implements AutoCloseable {

        private static final String PREFILL_ADDRESS = "127.0.0.1:18080";
        private static final String DECODE_ADDRESS = "127.0.0.1:18081";
        private static final long EXTERNAL_REQUEST_ID = 9_001L;

        private final FlexlbConfig config = config();
        private final ConfigService configService = new ConfigService() {
            @Override
            public FlexlbConfig loadBalanceConfig() {
                return config;
            }
        };
        private final RecordingSubmissionPort submission =
                new RecordingSubmissionPort();
        private final RequestSchedulerTestRuntime runtime;
        private final WorkerStatus prefillStatus;
        private final WorkerStatus decodeStatus;
        private final PrefillEndpoint prefill;
        private final DecodeEndpoint decode;
        private final PrefillGenerationRuntime.PreparedOffer
                externalPrefillCapacity;

        private Fixture(RoleType saturatedRole) {
            runtime = new RequestSchedulerTestRuntime(
                    configService,
                    submission,
                    new BatchSchedulerReporter(new NoOpFlexMonitor()),
                    new RequestSchedulerReporter(new NoOpFlexMonitor()),
                    new NoCancelChannel());

            prefillStatus = initializedStatus(
                    RoleType.PREFILL,
                    "127.0.0.1",
                    18_080);
            decodeStatus = initializedStatus(
                    RoleType.DECODE,
                    "127.0.0.1",
                    18_081);
            prefill = (PrefillEndpoint) runtime.endpointRegistry()
                    .registerPreinitializedEndpoint(
                            RoleType.PREFILL,
                            PREFILL_ADDRESS,
                            prefillStatus);
            decode = (DecodeEndpoint) runtime.endpointRegistry()
                    .registerPreinitializedEndpoint(
                            RoleType.DECODE,
                            DECODE_ADDRESS,
                            decodeStatus);
            if (saturatedRole == RoleType.PREFILL) {
                externalPrefillCapacity = holdPrefillBindingSeat(prefill);
            } else {
                externalPrefillCapacity = null;
                runtime.applyStatus(
                        decodeStatus,
                        statusResponse(RoleType.DECODE, 2L, true));
            }

            EngineWorkerStatus workers =
                    new EngineWorkerStatus(runtime.endpointRegistry());
            ResourceMeasureFactory measures = new ResourceMeasureFactory(
                    List.of(
                            new PrefillResourceMeasure(configService),
                            new DecodeResourceMeasure(configService)));
            ConfiguredLoadBalanceSelector selector =
                    new ConfiguredLoadBalanceSelector(List.of(
                            new RandomStrategy(workers, measures),
                            new CostBasedDecodeStrategy(workers, measures)));
            runtime.bindRouter(new DefaultRouter(
                    selector,
                    ignored -> GroupRoutingDecision.none(),
                    modelMeta()));
        }

        private BalanceContext context(long requestId) {
            return context(requestId, 50);
        }

        private BalanceContext context(long requestId, int priority) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setSeqLen(128L);
            request.setMaxNewTokens(8);
            request.setPriority(priority);
            request.setModel("capacity-contract-test");
            BalanceContext context = new BalanceContext();
            context.setRequest(request);
            context.setConfig(config);
            return context;
        }

        private void releaseExternalCapacity(RoleType role) {
            if (role == RoleType.PREFILL) {
                externalPrefillCapacity.close();
                return;
            }
            WorkerStatus status = role == RoleType.PREFILL
                    ? prefillStatus : decodeStatus;
            WorkerStatusResponse response = statusResponse(
                    role, status.appliedStatusCursor().statusVersion() + 1L,
                    false);
            runtime.applyStatus(status, response);
        }

        private void releaseExternalCapacityRepeatedly(
                RoleType role, int repetitions) throws Exception {
            WorkerStatus status = role == RoleType.PREFILL
                    ? prefillStatus : decodeStatus;
            WorkerStatusResponse response = statusResponse(
                    role,
                    status.appliedStatusCursor().statusVersion() + 1L,
                    false);
            CyclicBarrier start = new CyclicBarrier(repetitions);
            try (ExecutorService executor =
                         Executors.newFixedThreadPool(repetitions)) {
                List<? extends Future<?>> signals = IntStream.range(0, repetitions)
                        .mapToObj(ignored -> executor.submit(() -> {
                            await(start);
                            if (role == RoleType.PREFILL) {
                                externalPrefillCapacity.close();
                            }
                            runtime.applyStatus(status, response);
                        }))
                        .toList();
                for (Future<?> signal : signals) {
                    signal.get(2, TimeUnit.SECONDS);
                }
            }
        }

        @Override
        public void close() {
            if (externalPrefillCapacity != null) {
                externalPrefillCapacity.close();
            }
            runtime.close();
        }
    }

    private static final class CacheAffinityFixture implements AutoCloseable {

        private static final String GROUP = "capacity-contract";
        private static final String HOT_IP = "127.0.0.2";
        private static final String COLD_IP = "127.0.0.3";
        private static final String DECODE_IP = "127.0.0.4";
        private static final int PREFILL_PORT = 18_080;
        private static final int DECODE_PORT = 18_081;
        private static final String HOT_ADDRESS = HOT_IP + ":" + PREFILL_PORT;

        private final FlexlbConfig config = cacheAffinityConfig();
        private final ConfigService configService = new ConfigService() {
            @Override
            public FlexlbConfig loadBalanceConfig() {
                return config;
            }
        };
        private final RecordingSubmissionPort submission =
                new RecordingSubmissionPort();
        private final RequestSchedulerTestRuntime runtime;
        private final WorkerStatus hotStatus;
        private final PrefillEndpoint hot;
        private final PrefillEndpoint cold;
        private final PrefillGenerationRuntime.PreparedOffer hotCapacity;

        private CacheAffinityFixture() {
            runtime = new RequestSchedulerTestRuntime(
                    configService,
                    submission,
                    new BatchSchedulerReporter(new NoOpFlexMonitor()),
                    new RequestSchedulerReporter(new NoOpFlexMonitor()),
                    new NoCancelChannel());

            hotStatus = initializedStatus(
                    RoleType.PREFILL, HOT_IP, PREFILL_PORT);
            WorkerStatus coldStatus = initializedStatus(
                    RoleType.PREFILL, COLD_IP, PREFILL_PORT);
            WorkerStatus decodeStatus = initializedStatus(
                    RoleType.DECODE, DECODE_IP, DECODE_PORT);
            hot = (PrefillEndpoint) runtime.endpointRegistry()
                    .registerPreinitializedEndpoint(
                            RoleType.PREFILL, HOT_ADDRESS, hotStatus);
            cold = (PrefillEndpoint) runtime.endpointRegistry()
                    .registerPreinitializedEndpoint(
                            RoleType.PREFILL,
                            COLD_IP + ":" + PREFILL_PORT,
                            coldStatus);
            runtime.endpointRegistry().registerPreinitializedEndpoint(
                    RoleType.DECODE,
                    DECODE_IP + ":" + DECODE_PORT,
                    decodeStatus);

            EngineWorkerStatus workers =
                    new EngineWorkerStatus(runtime.endpointRegistry());
            ResourceMeasureFactory measures = new ResourceMeasureFactory(
                    List.of(
                            new PrefillResourceMeasure(configService),
                            new DecodeResourceMeasure(configService)));
            ConfiguredLoadBalanceSelector selector =
                    new ConfiguredLoadBalanceSelector(List.of(
                            new RandomStrategy(workers, measures),
                            new CostBasedPrefillStrategy(
                                    workers,
                                    new FixedCacheMatches(
                                            HOT_ADDRESS,
                                            COLD_IP + ":" + PREFILL_PORT),
                                    measures,
                                    mock(EngineHealthReporter.class,
                                            withSettings().stubOnly())),
                            new CostBasedDecodeStrategy(workers, measures)));
            runtime.bindRouter(new DefaultRouter(
                    selector,
                    ignored -> GroupRoutingDecision.none(),
                    modelMeta()));

            try (SelectedRole selected = selector.select(
                    context(300L), RoleType.PREFILL, GROUP)) {
                if (selected == null) {
                    throw new AssertionError(
                            "cache-affinity fixture has no Prefill selection");
                }
                assertEquals(HOT_IP, selected.serverStatus().getServerIp(),
                        "fixture must prove the hot endpoint is the cache leader");
                assertEquals(
                        896L,
                        selected.serverStatus().getDebugInfo().getHitCacheLen(),
                        "fixture must consume the exact-generation cache match");
            }
            hotCapacity = holdPrefillBindingSeat(hot);
        }

        private BalanceContext context(long requestId) {
            return context(requestId, false);
        }

        private BalanceContext coldContext(long requestId) {
            return context(requestId, true);
        }

        private BalanceContext context(long requestId, boolean coldAffinity) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setSeqLen(1_024L);
            request.setMaxNewTokens(8);
            request.setPriority(50);
            request.setModel("capacity-contract-test");
            request.setBlockCacheKeys(coldAffinity
                    ? List.of(11L, 12L, 13L, 14L, 15L, 16L, 17L, 18L)
                    : List.of(1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L));
            request.setCacheKeyBlockSize(128L);
            BalanceContext context = new BalanceContext();
            context.setRequest(request);
            context.setConfig(config);
            return context;
        }

        private void releaseHotCapacity() {
            hotCapacity.close();
        }

        @Override
        public void close() {
            hotCapacity.close();
            runtime.close();
        }
    }

    /** Hold the real per-generation bind seat without inventing Engine work. */
    private static PrefillGenerationRuntime.PreparedOffer
            holdPrefillBindingSeat(PrefillEndpoint endpoint) {
        try (WorkerEndpoint.GenerationPin pin = endpoint.tryPinGeneration()) {
            if (pin == null) {
                throw new AssertionError(
                        "fixture Prefill generation is not pinnable");
            }
            PrefillGenerationRuntime.PreparedOffer offer =
                    endpoint.prepareOfferPinned(
                            pin, Fixture.EXTERNAL_REQUEST_ID,
                            Integer.MAX_VALUE);
            if (offer == null) {
                throw new AssertionError(
                        "fixture could not hold the Prefill binding seat");
            }
            if (!offer.seal()) {
                offer.close();
                throw new AssertionError(
                        "fixture Prefill binding seat could not be sealed");
            }
            return offer;
        }
    }

    private static FlexlbConfig config() {
        FlexlbConfig config = baseConfig();
        config.getRouter().getRoles().getPrefill()
                .setSelector(new RoutingConfig.RandomPrefillSelectorConfig());
        return config;
    }

    private static FlexlbConfig cacheAffinityConfig() {
        FlexlbConfig config = baseConfig();
        RoutingConfig.CacheAffinityConfig affinity =
                new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(1_000L);
        affinity.setMinPrefixHitPercent(50.0);
        config.getRouter().getRoles().getPrefill()
                .setCacheAffinity(affinity);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(2L);
        return config;
    }

    private static FlexlbConfig baseConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.queueScheduler().setOrdering(new PriorityOrderingConfig());
        config.fixedWindowDecision().setMaxRequests(1);
        config.fixedWindowDecision().setMaxCollectionWaitMs(1L);
        config.queueScheduler().getCapacity()
                .setMaxOutstandingRequestsGlobal(64);
        config.queueScheduler().getCapacity()
                .setMaxWaitingRequestsPerPrefillWorker(8);
        BatchDispatcherConfig dispatcher =
                (BatchDispatcherConfig) config.getDispatcher();
        dispatcher.setMaxInflightBatchesPerPrefillWorker(2);
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(1L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(1L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxKvUsagePercent(100L);
        return config;
    }

    private static ModelMetaConfig modelMeta() {
        ModelMetaConfig modelMeta = mock(
                ModelMetaConfig.class, withSettings().stubOnly());
        when(modelMeta.requiredRoles()).thenReturn(
                List.of(RoleType.DECODE, RoleType.PREFILL));
        return modelMeta;
    }

    private static WorkerStatus initializedStatus(
            RoleType role,
            String ip,
            int httpPort) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                role, "capacity-contract", ip, httpPort, httpPort + 1, null);
        WorkerStatusResponse response = statusResponse(role, 1L, false);
        status.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared = status.prepareNewStatus(
                    status.freezeStatusResponse(response));
            status.publishPreparedStatus(prepared);
            status.recordSuccessfulPoll(true);
        } finally {
            status.lock.unlock();
        }
        return status;
    }

    private static WorkerStatusResponse statusResponse(
            RoleType role,
            long statusVersion,
            boolean saturated) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(statusVersion);
        response.setLatestFinishedVersion(0L);
        response.setAvailableKvCacheTokens(1_000_000L);
        response.setTotalKvCacheTokens(2_000_000L);
        response.setMaxSeqLen(1_000_000L);
        response.setMaxBatchTokensSize(1_000_000L);
        if (saturated) {
            TaskPhase phase = role == RoleType.PREFILL
                    ? TaskPhase.PENDING : TaskPhase.RUNNING;
            response.setRunningTaskInfo(Map.of(
                    Long.toString(Fixture.EXTERNAL_REQUEST_ID),
                    task(Fixture.EXTERNAL_REQUEST_ID, phase)));
            if (role == RoleType.PREFILL) {
                response.setWaitingQueryLen(1L);
            } else {
                response.setRunningQueryLen(1L);
            }
        } else {
            response.setRunningTaskInfo(Map.of());
        }
        response.setFinishedTaskInfo(Map.of());
        return response;
    }

    private static TaskInfo task(long requestId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setPhase(phase);
        task.setInputLength(128L);
        return task;
    }

    private static String prefillIp(Response response) {
        return response.getServerStatus().stream()
                .filter(status -> status.getRole() == RoleType.PREFILL)
                .map(status -> status.getServerIp())
                .findFirst()
                .orElse("<none>");
    }

    private static final class FixedCacheMatches
            implements CacheAwareService {

        private final String hotAddress;
        private final String coldAddress;

        private FixedCacheMatches(String hotAddress, String coldAddress) {
            this.hotAddress = hotAddress;
            this.coldAddress = coldAddress;
        }

        @Override
        public Map<EngineGeneration, CacheMatch> findMatchingEngines(
                List<Long> blockCacheKeys,
                RoleType roleType,
                List<EngineGeneration> candidates) {
            if (roleType != RoleType.PREFILL || blockCacheKeys.isEmpty()) {
                return Map.of();
            }
            String leader = blockCacheKeys.get(0) >= 10L
                    ? coldAddress : hotAddress;
            return candidates.stream()
                    .filter(candidate -> leader.equals(candidate.address()))
                    .collect(java.util.stream.Collectors.toUnmodifiableMap(
                            candidate -> candidate,
                            candidate -> new CacheMatch(blockCacheKeys.size())));
        }

        @Override
        public boolean activateEngineGeneration(
                String engineIpPort,
                RoleType roleType,
                long generationId) {
            throw new AssertionError(
                    "fixed query stub does not own cache lifecycle");
        }

        @Override
        public WorkerCacheUpdateResult updateEngineBlockCache(
                String engineIpPort,
                RoleType roleType,
                long generationId,
                org.flexlb.dao.master.CacheStatus cacheStatus) {
            throw new AssertionError(
                    "fixed query stub does not own cache lifecycle");
        }

        @Override
        public boolean retireEngineGeneration(
                String engineIpPort,
                RoleType roleType,
                long generationId) {
            throw new AssertionError(
                    "fixed query stub does not own cache lifecycle");
        }
    }

    private static final class RecordingSubmissionPort
            implements BatchSubmissionPort {

        private final List<Command> commands = new CopyOnWriteArrayList<>();
        private final Semaphore commandSignals = new Semaphore(0);
        private final AtomicBoolean holdCompletions = new AtomicBoolean();

        @Override
        public CapacityBoundary.Attempt<PreparedSubmission> prepare() {
            return new CapacityBoundary.Attempt.Accepted<>(
                    new PreparedSubmission() {
                        private final AtomicBoolean submitted =
                                new AtomicBoolean();

                        @Override
                        public void submit(
                                Command command,
                                Observer observer) {
                            if (!submitted.compareAndSet(false, true)) {
                                throw new IllegalStateException(
                                        "prepared submission reused");
                            }
                            commands.add(command);
                            commandSignals.release();
                            for (DeliveryItem item : command.exactItems()) {
                                if (!holdCompletions.get()) {
                                    observer.onCompletion(
                                            item,
                                            SlotDeliveryPort.Completion.Delivered.INSTANCE);
                                }
                            }
                        }

                        @Override
                        public void close() {
                            // An unused preparation owns no external resource.
                        }
                    });
        }

        private List<Command> commands() {
            return List.copyOf(commands);
        }

        private List<Long> requestIds() {
            return commands.stream()
                    .flatMap(command -> command.exactItems().stream())
                    .map(DeliveryItem::requestId)
                    .toList();
        }

        private void holdCompletions() {
            holdCompletions.set(true);
        }

        private boolean awaitCommands(
                int count, long timeout, TimeUnit unit)
                throws InterruptedException {
            return commandSignals.tryAcquire(count, timeout, unit);
        }

        private String prefillIp(long requestId) {
            return commands.stream()
                    .flatMap(command -> command.exactItems().stream())
                    .filter(item -> item.requestId() == requestId)
                    .map(item -> ((BatchItem) item).prefillEp().getIp())
                    .findFirst()
                    .orElse("<none>");
        }
    }

    private static void await(CyclicBarrier barrier) {
        try {
            barrier.await(2, TimeUnit.SECONDS);
        } catch (Exception failure) {
            throw new AssertionError("concurrent test barrier failed", failure);
        }
    }

    private static final class NoCancelChannel
            implements EngineCancelChannel {

        @Override
        public boolean isSupported(DecodeEndpoint endpoint) {
            return false;
        }

        @Override
        public CompletableFuture<CancelOutcome> cancel(
                org.flexlb.balance.preemption.CancelTarget target,
                long requestId,
                long timeoutMs) {
            return CompletableFuture.completedFuture(
                    CancelOutcome.unsupported());
        }
    }
}
