package org.flexlb.balance.scheduler;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/** External contract for temporary P/D capacity pressure. */
class TransientCapacityQueueContractTest {

    private static final int INCIDENT_PENDING_REQUESTS = 10_000;
    private static final int INCIDENT_PREFILL_ENGINES = 750;
    private static final int INCIDENT_DECODE_ENGINES = 750;
    private static final int INCIDENT_DECODE_MAX_CONCURRENCY = 128;
    private static final int INCIDENT_WORKER_STATUS_UPDATES = 1_000;

    @Test
    @Timeout(20)
    void deferredDecodeRetryWaitsForSettlementFenceToExpire()
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        config.setDispatcher(DispatcherConfig.nonBatch());
        verifyRetryWaitsForDecodeSettlementFence(config, 880_001L);
    }

    @Test
    @Timeout(20)
    void capacityCheckedDecodeRetryWaitsForSettlementFenceToExpire()
            throws Exception {
        FlexlbConfig config = config();
        ((QueueOrderingConfig) config.queueScheduler().getOrdering())
                .setPreemption(new PreemptionConfig());
        verifyRetryWaitsForDecodeSettlementFence(config, 880_002L);
    }

    private static void verifyRetryWaitsForDecodeSettlementFence(
            FlexlbConfig config,
            long requestId) throws Exception {
        try (Fixture fixture = new Fixture(null, config)) {
            DecodeEndpoint.ReservationHandle settled;
            try (WorkerEndpoint.GenerationPin pin =
                         fixture.decodeEndpoint.tryPinGeneration()) {
                settled = fixture.decodeEndpoint.reserveQueuedPinned(
                        pin, requestId, 128L, 136L, 50);
            }
            assertTrue(fixture.decodeEndpoint.releaseLocalShadowIfExact(
                    settled));
            assertEquals(0, fixture.totalDecodeReservations());

            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    statusResponse(RoleType.PREFILL, 2L, true));
            CompletableFuture<Response> retried =
                    fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50, 128_000L));
            assertFalse(retried.isDone());

            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    statusResponse(RoleType.PREFILL, 3L, false));

            Thread.sleep(100L);
            assertFalse(retried.isDone(),
                    "a live Decode settlement fence is a wait, not a failure");
            Thread.sleep(2L);
            fixture.decodeEndpoint.evictExpiredRequests(
                    0L, ignored -> false);

            assertTrue(retried.get(5, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test
    @Timeout(90)
    void workerStatusHeartbeatsCannotReplayTenThousandPendingPlacements()
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().getCapacity()
                .setMaxOutstandingRequestsGlobal(
                        INCIDENT_PENDING_REQUESTS + 1);
        config.queueScheduler().getCapacity()
                .setMaxWaitingRequestsPerPrefillWorker(
                        INCIDENT_PENDING_REQUESTS + 1);
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests((long) INCIDENT_PENDING_REQUESTS + 1L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests((long) INCIDENT_DECODE_MAX_CONCURRENCY);
        ((QueueOrderingConfig) config.queueScheduler().getOrdering())
                .setPreemption(new PreemptionConfig());

        try (QuietPlacementLogs ignored = new QuietPlacementLogs();
             Fixture fixture = new Fixture(RoleType.DECODE, config, true)) {
            fixture.expandFleet(
                    INCIDENT_PREFILL_ENGINES,
                    INCIDENT_DECODE_ENGINES,
                    INCIDENT_DECODE_MAX_CONCURRENCY);
            assertEquals(INCIDENT_PREFILL_ENGINES,
                    fixture.runtime.endpointRegistry()
                            .getEndpointCount(RoleType.PREFILL));
            assertEquals(INCIDENT_DECODE_ENGINES,
                    fixture.runtime.endpointRegistry()
                            .getEndpointCount(RoleType.DECODE));
            fixture.metrics.resetPlacementWakeups();
            fixture.submission.holdCompletions();
            long firstRequestId = 100_000L;
            long absoluteDeadline = System.currentTimeMillis()
                    + TimeUnit.MINUTES.toMillis(5);
            List<BalanceContext> contexts =
                    new ArrayList<>(INCIDENT_PENDING_REQUESTS);
            List<CompletableFuture<Response>> futures =
                    new ArrayList<>(INCIDENT_PENDING_REQUESTS);

            for (int index = 0; index < INCIDENT_PENDING_REQUESTS; index++) {
                BalanceContext context = fixture.context(
                        firstRequestId + index, 50, 128_000L);
                context.setSchedulingMetadata(SchedulingMetadata.explicit(
                        50, absoluteDeadline));
                contexts.add(context);
                futures.add(fixture.runtime.scheduler().submit(context));
            }

            assertEquals(INCIDENT_PENDING_REQUESTS,
                    fixture.runtime.scheduler().getQueuedRequestCount());
            assertEquals(INCIDENT_PENDING_REQUESTS,
                    fixture.metrics.totalPlacementAttempts());
            assertTrue(fixture.metrics.candidateEvaluations()
                            >= (long) INCIDENT_PENDING_REQUESTS
                                    * INCIDENT_DECODE_ENGINES,
                    "the incident setup did not scan the full Decode fleet");
            assertEquals(0, fixture.totalPrefillQueuedRequests(),
                    "P-success/D-failure must not retain a Prefill queue seat");
            assertEquals(0,
                    fixture.totalPrefillOwnedRequests(),
                    "P-success/D-failure must not retain Prefill ownership");
            assertEquals(0,
                    fixture.totalDecodeReservations(),
                    "P-success/D-failure must not retain a Decode reservation");
            assertCanonicalWaitState(
                    fixture, contexts, futures, absoluteDeadline);

            int attemptsBeforeStatus =
                    fixture.metrics.totalPlacementAttempts();
            long evaluationsBeforeStatus =
                    fixture.metrics.candidateEvaluations();
            fixture.publishDecodeStatusStorm(
                    INCIDENT_WORKER_STATUS_UPDATES,
                    INCIDENT_DECODE_MAX_CONCURRENCY);
            Thread.sleep(100L);

            int statusTriggeredPlacements =
                    fixture.metrics.totalPlacementAttempts()
                            - attemptsBeforeStatus;
            assertEquals(0, statusTriggeredPlacements,
                    "WorkerStatus heartbeat replayed pending placement");
            assertEquals(evaluationsBeforeStatus,
                    fixture.metrics.candidateEvaluations(),
                    "WorkerStatus heartbeat evaluated placement candidates");
            assertEquals(0L, fixture.metrics.placementWakeups(),
                    "unchanged WorkerStatus must not publish a capacity edge");
            assertEquals(INCIDENT_PENDING_REQUESTS,
                    fixture.runtime.scheduler().getQueuedRequestCount());
            assertCanonicalWaitState(
                    fixture, contexts, futures, absoluteDeadline);

            long capacityReleasedAt = System.nanoTime();
            fixture.runtime.applyStatus(
                    fixture.decodeStatus,
                    decodeRunningStatus(
                            fixture.decodeStatus.appliedStatusCursor()
                                    .statusVersion() + 1L,
                            INCIDENT_DECODE_MAX_CONCURRENCY - 1));

            assertTrue(fixture.submission.awaitCommands(
                    1, 5, TimeUnit.SECONDS));
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
            assertTrue(awaitPlacementAttempts(
                    fixture.metrics, attemptsBeforeStatus + 2, 5_000));
            assertTrue(awaitPlacementQuiescence(
                    fixture.metrics, 100L, 5_000L));
            List<Long> capacityRetryOrder =
                    fixture.metrics.requestIdsFrom(attemptsBeforeStatus);
            assertFalse(capacityRetryOrder.isEmpty());
            assertEquals(firstRequestId, capacityRetryOrder.getFirst(),
                    "the original placement sequence must survive the wait");
            assertEquals(1, fixture.submission.requestIds().size(),
                    "one Decode slot must not be oversold across Prefill batchers");
            assertTrue(capacityRetryOrder.contains(
                            fixture.submission.requestIds().getFirst()),
                    "the dispatched request must come from this capacity round");
            int capacityTriggeredPlacements =
                    fixture.metrics.totalPlacementAttempts()
                            - attemptsBeforeStatus;
            assertTrue(capacityTriggeredPlacements <= 2,
                    "one Decode slot retried beyond its ordered capacity boundary: "
                            + capacityTriggeredPlacements
                            + ", capacity edges="
                            + fixture.metrics.placementWakeups()
                            + ", retry order=" + capacityRetryOrder);
            assertTrue(fixture.metrics.maxAttemptsForAnyRequest() <= 2,
                    "one capacity opportunity retried one request more than once");
            assertEquals(1L, fixture.metrics.placementWakeups(),
                    "one real capacity release must publish one placement edge");
            assertCanonicalWaitState(
                    fixture, contexts, futures, absoluteDeadline);

            fixture.metrics.print(
                    "worker-status-and-capacity-edge",
                    INCIDENT_PENDING_REQUESTS,
                    fixture.runtime.scheduler().getQueuedRequestCount(),
                    statusTriggeredPlacements,
                    capacityReleasedAt);
        }
    }

    private static void assertCanonicalWaitState(
            Fixture fixture,
            List<BalanceContext> contexts,
            List<CompletableFuture<Response>> futures,
            long absoluteDeadline) {
        for (int index = 0; index < contexts.size(); index++) {
            BalanceContext context = contexts.get(index);
            assertSame(futures.get(index),
                    fixture.runtime.requestFuture(context.getRequestId()),
                    "waiting must retain the canonical future");
            assertSame(context, fixture.metrics.context(context.getRequestId()),
                    "placement retry replaced the BalanceContext");
            assertEquals(absoluteDeadline,
                    context.schedulingMetadata().expiresAtMs(),
                    "placement retry changed the absolute deadline");
            assertEquals(absoluteDeadline,
                    fixture.metrics.deadline(context.getRequestId()),
                    "placement observed a different absolute deadline");
        }
    }

    private static boolean awaitPlacementAttempts(
            PlacementMetrics metrics, int expected, long timeoutMs)
            throws InterruptedException {
        long deadline = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        while (metrics.totalPlacementAttempts() < expected
                && System.nanoTime() < deadline) {
            Thread.sleep(1L);
        }
        return metrics.totalPlacementAttempts() >= expected;
    }

    private static boolean awaitPlacementQuiescence(
            PlacementMetrics metrics,
            long stableForMs,
            long timeoutMs) throws InterruptedException {
        long timeoutAt = System.nanoTime()
                + TimeUnit.MILLISECONDS.toNanos(timeoutMs);
        int observed = metrics.totalPlacementAttempts();
        long unchangedSince = System.nanoTime();
        while (System.nanoTime() < timeoutAt) {
            Thread.sleep(1L);
            int current = metrics.totalPlacementAttempts();
            if (current != observed) {
                observed = current;
                unchangedSince = System.nanoTime();
            } else if (System.nanoTime() - unchangedSince
                    >= TimeUnit.MILLISECONDS.toNanos(stableForMs)) {
                return true;
            }
        }
        return false;
    }

    @Test
    @Timeout(20)
    void prefillPressureRetainsAndLaterDispatchesTheOriginalRequest()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(fixture.context(101L));

            assertFalse(waiting.isDone());
            assertEquals(1, fixture.runtime.scheduler().getQueuedRequestCount());
            assertEquals(List.of(), fixture.submission.requestIds());

            fixture.releaseCapacity();

            assertTrue(waiting.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(List.of(101L), fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void decodePressureRetainsAndLaterDispatchesTheOriginalRequest()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(fixture.context(201L));

            assertFalse(waiting.isDone());
            assertEquals(List.of(), fixture.submission.requestIds());

            fixture.releaseCapacity();

            assertTrue(waiting.get(2, TimeUnit.SECONDS).isSuccess());
            assertEquals(List.of(201L), fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void releasedSeatPreservesFifoAtTheSamePriority() throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            fixture.submission.holdCompletions();
            CompletableFuture<Response> older =
                    fixture.runtime.scheduler().submit(
                            fixture.context(251L, 50));
            CompletableFuture<Response> later =
                    fixture.runtime.scheduler().submit(
                            fixture.context(252L, 50));

            assertFalse(older.isDone());
            assertFalse(later.isDone());
            fixture.releaseCapacity();

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertEquals(List.of(251L), fixture.submission.requestIds());
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    @Timeout(20)
    void releasedSeatUsesPriorityOrderingBeforeArrivalOrder()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.PREFILL)) {
            fixture.submission.holdCompletions();
            fixture.runtime.scheduler().submit(fixture.context(261L, 50));
            fixture.runtime.scheduler().submit(fixture.context(262L, 80));

            fixture.releaseCapacity();

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertEquals(List.of(262L), fixture.submission.requestIds());
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    @Timeout(20)
    void releasedDecodeCapacityPreservesFifo() throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            fixture.submission.holdCompletions();
            fixture.runtime.scheduler().submit(fixture.context(271L, 50));
            fixture.runtime.scheduler().submit(fixture.context(272L, 50));

            fixture.releaseCapacity();

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertEquals(List.of(271L), fixture.submission.requestIds());
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
        }
    }

    @Test
    @Timeout(20)
    void duplicateCapacitySignalsCannotOversellOneDecodeSlot()
            throws Exception {
        try (Fixture fixture = new Fixture(RoleType.DECODE)) {
            fixture.submission.holdCompletions();
            fixture.runtime.scheduler().submit(fixture.context(281L, 50));
            fixture.runtime.scheduler().submit(fixture.context(282L, 50));

            fixture.releaseCapacity();
            fixture.runtime.applyStatus(
                    fixture.decodeStatus,
                    statusResponse(RoleType.DECODE, 4L, false));

            assertTrue(fixture.submission.awaitCommands(
                    1, 2, TimeUnit.SECONDS));
            assertFalse(fixture.submission.awaitCommands(
                    1, 200, TimeUnit.MILLISECONDS));
            assertEquals(List.of(281L), fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void nonPreemptiveLongContextBacklogQueuesBeforeDecodeCapacityReturns()
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        config.queueScheduler().getCapacity()
                .setMaxOutstandingRequestsGlobal(64);
        config.queueScheduler().getCapacity()
                .setMaxWaitingRequestsPerPrefillWorker(64);
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(64L);

        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            List<CompletableFuture<Response>> waiting = LongStream
                    .rangeClosed(301L, 332L)
                    .mapToObj(requestId -> fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50, 128_000L)))
                    .toList();

            assertTrue(waiting.stream().noneMatch(CompletableFuture::isDone));
            assertEquals(32,
                    fixture.decodeEndpoint.layeredAdmissionView().queued().size(),
                    "transient Decode pressure must not strand long contexts"
                            + " in the global placement coordinator");
            assertEquals(List.of(), fixture.submission.requestIds());
        }
    }

    @Test
    @Timeout(20)
    void incidentBacklogNeverPinsToFullDecodeWhileAnotherCanDispatch()
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        config.queueScheduler().getCapacity()
                .setMaxOutstandingRequestsGlobal(64);
        config.queueScheduler().getCapacity()
                .setMaxWaitingRequestsPerPrefillWorker(64);
        // The fixture's external PENDING sentinel consumes one Prefill
        // admission count; keep room for the production-local bound of 64.
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(65L);
        config.getRouter().getRoles().getDecode().setSelector(
                new RoutingConfig.KvUsageWeightedRandomConfig());
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(64L);
        assertTrue(config.defersDecodeCapacityUntilDispatch());

        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            fixture.submission.holdCompletions();
            fixture.runtime.applyStatus(
                    fixture.decodeStatus, decodeRunningStatus(4L, 64));
            fixture.runtime.applyStatus(
                    fixture.prefillStatus,
                    statusResponse(RoleType.PREFILL, 3L, true));
            WorkerStatus spareStatus = initializedStatus(
                    RoleType.DECODE, "127.0.0.2", 18_082);
            DecodeEndpoint spareEndpoint = (DecodeEndpoint)
                    fixture.runtime.endpointRegistry()
                            .registerPreinitializedEndpoint(
                                    RoleType.DECODE,
                                    "127.0.0.2:18082",
                                    spareStatus);
            assertEquals(64, fixture.decodeEndpoint.getEngineLoad());
            assertEquals(0, spareEndpoint.getEngineLoad());
            assertFalse(fixture.decodeMeasure.isResourceAvailable(
                    fixture.decodeEndpoint.routingView()));
            assertTrue(fixture.decodeMeasure.isResourceAvailable(
                    spareEndpoint.routingView()));

            List<CompletableFuture<Response>> waiting = LongStream
                    .rangeClosed(401L, 464L)
                    .mapToObj(requestId -> fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50, 128_000L)))
                    .toList();

            assertTrue(waiting.stream().noneMatch(CompletableFuture::isDone));
            assertEquals(0,
                    fixture.decodeEndpoint.layeredAdmissionView().reserved().size(),
                    "the incident's full Decode must not own a Prefill queue head");
            assertEquals(64,
                    spareEndpoint.layeredAdmissionView().reserved().size(),
                    "all production-bound waiting requests should pin the dispatchable tier");
        }
    }

    @Test
    @Timeout(20)
    void queuedHeadRebindsWhenItsDecodeFillsBeforeDispatch()
            throws Exception {
        FlexlbConfig config = config();
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        assertTrue(config.defersDecodeCapacityUntilDispatch());

        try (Fixture fixture = new Fixture(null, config)) {
            fixture.submission.blockPreparation();
            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(
                            fixture.context(501L, 50, 128_000L));
            assertTrue(fixture.submission.awaitPreparation(
                    2, TimeUnit.SECONDS));

            fixture.runtime.applyStatus(
                    fixture.decodeStatus,
                    statusResponse(RoleType.DECODE, 2L, true));
            WorkerStatus spareStatus = initializedStatus(
                    RoleType.DECODE, "127.0.0.2", 18_082);
            DecodeEndpoint spareEndpoint = (DecodeEndpoint)
                    fixture.runtime.endpointRegistry()
                            .registerPreinitializedEndpoint(
                                    RoleType.DECODE,
                                    "127.0.0.2:18082",
                                    spareStatus);

            fixture.submission.releasePreparation();

            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals("127.0.0.2:18082", decodeAddress(response));
            assertEquals(List.of("127.0.0.2:18082"),
                    fixture.submission.decodeAddresses());
            assertEquals(0,
                    fixture.decodeEndpoint.layeredAdmissionView()
                            .reserved().size(),
                    "the stale Decode binding must release its queued hold");
            assertEquals(1, spareEndpoint.getEngineLoad(),
                    "the replacement Decode must own the dispatched request");
        }
    }

    @Test
    @Timeout(20)
    void queuedHeadWakesWhenAnotherDecodeReturnsCapacity()
            throws Exception {
        verifyPoolCapacityWake(config(), 601L);
    }

    @Test
    @Timeout(20)
    void nonBatchHeadAlsoWakesAndRebindsAcrossTheDecodePool()
            throws Exception {
        FlexlbConfig config = config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        verifyPoolCapacityWake(config, 602L);
    }

    private static void verifyPoolCapacityWake(
            FlexlbConfig config,
            long requestId) throws Exception {
        config.queueScheduler().setOrdering(new QueueOrderingConfig());
        assertTrue(config.defersDecodeCapacityUntilDispatch());

        try (Fixture fixture = new Fixture(RoleType.DECODE, config)) {
            WorkerStatus spareStatus = initializedStatus(
                    RoleType.DECODE, "127.0.0.2", 18_082);
            DecodeEndpoint spareEndpoint = (DecodeEndpoint)
                    fixture.runtime.endpointRegistry()
                            .registerPreinitializedEndpoint(
                                    RoleType.DECODE,
                                    "127.0.0.2:18082",
                                    spareStatus);
            fixture.runtime.applyStatus(
                    spareStatus,
                    statusResponse(RoleType.DECODE, 2L, true));

            CompletableFuture<Response> waiting =
                    fixture.runtime.scheduler().submit(
                            fixture.context(requestId, 50, 128_000L));
            assertFalse(waiting.isDone());

            boolean boundToPrimary = fixture.decodeEndpoint
                    .layeredAdmissionView().reserved().size() == 1;
            WorkerStatus releasedStatus = boundToPrimary
                    ? spareStatus : fixture.decodeStatus;
            String releasedAddress = boundToPrimary
                    ? "127.0.0.2:18082" : "127.0.0.1:18081";
            fixture.runtime.applyStatus(
                    releasedStatus,
                    statusResponse(RoleType.DECODE, 3L, false));

            Response response = waiting.get(2, TimeUnit.SECONDS);
            assertTrue(response.isSuccess());
            assertEquals(releasedAddress, decodeAddress(response));
            DecodeEndpoint releasedEndpoint = boundToPrimary
                    ? spareEndpoint : fixture.decodeEndpoint;
            assertEquals(1, releasedEndpoint.getEngineLoad());
        }
    }

    private static String decodeAddress(Response response) {
        return response.getServerStatus().stream()
                .filter(status -> status.getRole() == RoleType.DECODE)
                .map(status -> status.getServerIp() + ":"
                        + status.getHttpPort())
                .findFirst()
                .orElseThrow();
    }

    private static final class Fixture implements AutoCloseable {
        private static final long EXTERNAL_REQUEST_ID = 9_001L;
        private static final String PREFILL_ADDRESS = "127.0.0.1:18080";
        private static final String DECODE_ADDRESS = "127.0.0.1:18081";

        private final FlexlbConfig config;
        private final ConfigService configService = new ConfigService() {
            @Override
            public FlexlbConfig loadBalanceConfig() {
                return config;
            }
        };
        private final RecordingSubmissionPort submission =
                new RecordingSubmissionPort();
        private final PlacementMetrics metrics = new PlacementMetrics();
        private final RequestSchedulerTestRuntime runtime;
        private final WorkerStatus prefillStatus;
        private final WorkerStatus decodeStatus;
        private final PrefillEndpoint prefillEndpoint;
        private final DecodeEndpoint decodeEndpoint;
        private final List<PrefillEndpoint> prefillEndpoints =
                new ArrayList<>();
        private final List<DecodeEndpoint> decodeEndpoints =
                new ArrayList<>();
        private final List<WorkerStatus> decodeStatuses =
                new ArrayList<>();
        private final DecodeResourceMeasure decodeMeasure;
        private final RoleType saturatedRole;

        private Fixture(RoleType saturatedRole) {
            this(saturatedRole, config());
        }

        private Fixture(RoleType saturatedRole, FlexlbConfig config) {
            this(saturatedRole, config, false);
        }

        private Fixture(
                RoleType saturatedRole,
                FlexlbConfig config,
                boolean prefillFirst) {
            this.saturatedRole = saturatedRole;
            this.config = config;
            runtime = new RequestSchedulerTestRuntime(
                    configService,
                    submission,
                    new BatchSchedulerReporter(new NoOpFlexMonitor()),
                    new RequestSchedulerReporter(new NoOpFlexMonitor()),
                    new NoCancelChannel());

            prefillStatus = initializedStatus(
                    RoleType.PREFILL, "127.0.0.1", 18_080);
            decodeStatus = initializedStatus(
                    RoleType.DECODE, "127.0.0.1", 18_081);
            prefillEndpoint = (PrefillEndpoint) runtime.endpointRegistry()
                    .registerPreinitializedEndpoint(
                            RoleType.PREFILL,
                            PREFILL_ADDRESS,
                            prefillStatus);
            decodeEndpoint = (DecodeEndpoint) runtime.endpointRegistry()
                    .registerPreinitializedEndpoint(
                    RoleType.DECODE,
                    DECODE_ADDRESS,
                    decodeStatus);
            prefillEndpoints.add(prefillEndpoint);
            decodeEndpoints.add(decodeEndpoint);
            decodeStatuses.add(decodeStatus);

            if (saturatedRole == RoleType.PREFILL) {
                runtime.applyStatus(
                        prefillStatus,
                        statusResponse(RoleType.PREFILL, 2L, true));
            } else if (saturatedRole == RoleType.DECODE) {
                runtime.applyStatus(
                        decodeStatus,
                        statusResponse(RoleType.DECODE, 2L, true));
            }

            WorkerDirectory workers =
                    new WorkerDirectory(runtime.endpointRegistry());
            CountingPrefillResourceMeasure prefillMeasure =
                    new CountingPrefillResourceMeasure(
                            configService, metrics);
            decodeMeasure = new CountingDecodeResourceMeasure(
                    configService, metrics);
            ConfiguredLoadBalanceSelector selector =
                    new ConfiguredLoadBalanceSelector(
                            List.of(
                                    new RandomStrategy(
                                            workers, prefillMeasure, decodeMeasure),
                                    new CostBasedDecodeStrategy(
                                            workers, decodeMeasure)));
            runtime.placementAvailability().addListener(
                    (key, ignoredSequence) ->
                            metrics.recordPlacementWakeup(key));
            runtime.bindRouter(new RecordingRouter(
                    new DefaultRouter(
                            selector,
                            ignored -> GroupRoutingDecision.none(),
                            modelMeta(prefillFirst),
                            runtime.placementAvailability()),
                    metrics));
        }

        private void expandFleet(
                int prefillCount,
                int decodeCount,
                int decodeRunningCount) {
            if (prefillCount < 1 || decodeCount < 1) {
                throw new IllegalArgumentException(
                        "fleet sizes must include the primary endpoints");
            }
            runtime.applyStatus(
                    decodeStatus,
                    decodeRunningStatus(
                            decodeStatus.appliedStatusCursor()
                                    .statusVersion() + 1L,
                            decodeRunningCount));
            for (int index = 1; index < prefillCount; index++) {
                String ip = fleetIp(1, index);
                int port = 20_080;
                WorkerStatus status = initializedStatus(
                        RoleType.PREFILL, ip, port);
                PrefillEndpoint endpoint = (PrefillEndpoint)
                        runtime.endpointRegistry()
                                .registerPreinitializedEndpoint(
                                        RoleType.PREFILL,
                                        ip + ":" + port,
                                        status);
                prefillEndpoints.add(endpoint);
            }
            for (int index = 1; index < decodeCount; index++) {
                String ip = fleetIp(2, index);
                int port = 20_081;
                WorkerStatus status = initializedStatus(
                        RoleType.DECODE, ip, port);
                DecodeEndpoint endpoint = (DecodeEndpoint)
                        runtime.endpointRegistry()
                                .registerPreinitializedEndpoint(
                                        RoleType.DECODE,
                                        ip + ":" + port,
                                        status);
                runtime.applyStatus(
                        status,
                        decodeRunningStatus(2L, decodeRunningCount));
                decodeStatuses.add(status);
                decodeEndpoints.add(endpoint);
            }
        }

        private void publishDecodeStatusStorm(
                int updateCount,
                int runningCount) throws Exception {
            if (updateCount < 0) {
                throw new IllegalArgumentException(
                        "update count cannot be negative");
            }
            int threads = Math.min(32, Math.max(1, decodeStatuses.size()));
            try (ExecutorService executor =
                         Executors.newFixedThreadPool(threads)) {
                int published = 0;
                while (published < updateCount) {
                    int waveSize = Math.min(
                            decodeStatuses.size(), updateCount - published);
                    List<Future<?>> wave = new ArrayList<>(waveSize);
                    for (int index = 0; index < waveSize; index++) {
                        WorkerStatus status = decodeStatuses.get(index);
                        long version = status.appliedStatusCursor()
                                .statusVersion() + 1L;
                        wave.add(executor.submit(() -> runtime.applyStatus(
                                status,
                                decodeRunningStatus(version, runningCount))));
                    }
                    for (Future<?> update : wave) {
                        update.get(10, TimeUnit.SECONDS);
                    }
                    published += waveSize;
                }
            }
        }

        private int totalPrefillQueuedRequests() {
            return prefillEndpoints.stream()
                    .mapToInt(PrefillEndpoint::queuedRequestCount)
                    .sum();
        }

        private int totalPrefillOwnedRequests() {
            return prefillEndpoints.stream()
                    .mapToInt(PrefillEndpoint::getLocallyOwnedRequestCount)
                    .sum();
        }

        private int totalDecodeReservations() {
            return decodeEndpoints.stream()
                    .mapToInt(endpoint -> endpoint.layeredAdmissionView()
                            .reserved().size())
                    .sum();
        }

        private static String fleetIp(int roleSubnet, int index) {
            return "127." + roleSubnet + "." + index / 250
                    + "." + (index % 250 + 1);
        }

        private BalanceContext context(long requestId) {
            return context(requestId, 50);
        }

        private BalanceContext context(long requestId, int priority) {
            return context(requestId, priority, 128L);
        }

        private BalanceContext context(
                long requestId, int priority, long sequenceLength) {
            Request request = new Request();
            request.setRequestId(requestId);
            request.setSeqLen(sequenceLength);
            request.setMaxNewTokens(8);
            request.setPriority(priority);
            request.setModel("transient-capacity-contract");
            BalanceContext context = new BalanceContext();
            context.setRequest(request);
            context.setConfig(config);
            return context;
        }

        private void releaseCapacity() {
            if (saturatedRole == RoleType.PREFILL) {
                runtime.applyStatus(
                        prefillStatus,
                        statusResponse(RoleType.PREFILL, 3L, false));
                return;
            }
            runtime.applyStatus(
                    decodeStatus,
                    statusResponse(RoleType.DECODE, 3L, false));
        }

        @Override
        public void close() {
            runtime.close();
        }
    }

    private static final class RecordingRouter implements Router {
        private final Router delegate;
        private final PlacementMetrics metrics;

        private RecordingRouter(Router delegate, PlacementMetrics metrics) {
            this.delegate = delegate;
            this.metrics = metrics;
        }

        @Override
        public Response routeDirect(BalanceContext context) {
            long startedAt = metrics.placementStarted(context);
            try {
                return delegate.routeDirect(context);
            } finally {
                metrics.placementFinished(startedAt);
            }
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            long startedAt = metrics.placementStarted(context);
            try {
                return delegate.routeForQueue(context);
            } finally {
                metrics.placementFinished(startedAt);
            }
        }
    }

    private static final class CountingPrefillResourceMeasure
            extends PrefillResourceMeasure {
        private final PlacementMetrics metrics;

        private CountingPrefillResourceMeasure(
                ConfigService configService, PlacementMetrics metrics) {
            super(configService);
            this.metrics = metrics;
        }

        @Override
        public boolean isResourceAvailable(long pendingRequests) {
            metrics.recordCandidateEvaluation();
            return super.isResourceAvailable(pendingRequests);
        }
    }

    private static final class CountingDecodeResourceMeasure
            extends DecodeResourceMeasure {
        private final PlacementMetrics metrics;

        private CountingDecodeResourceMeasure(
                ConfigService configService, PlacementMetrics metrics) {
            super(configService);
            this.metrics = metrics;
        }

        @Override
        public boolean isResourceAvailable(
                DecodeEndpoint.DecodeRoutingView view) {
            metrics.recordCandidateEvaluation();
            return super.isResourceAvailable(view);
        }

        @Override
        public boolean isEngineDispatchAvailable(
                DecodeEndpoint.DecodeRoutingView view) {
            metrics.recordCandidateEvaluation();
            return super.isEngineDispatchAvailable(view);
        }
    }

    private static final class PlacementMetrics {
        private final AtomicInteger totalAttempts = new AtomicInteger();
        private final AtomicLong candidateEvaluations = new AtomicLong();
        private final AtomicLong placementWakeups = new AtomicLong();
        private final Map<Long, AtomicInteger> attemptsByRequest =
                new ConcurrentHashMap<>();
        private final Map<Long, BalanceContext> contexts =
                new ConcurrentHashMap<>();
        private final Map<Long, Long> deadlines = new ConcurrentHashMap<>();
        private final List<Long> placementLatenciesNanos =
                new CopyOnWriteArrayList<>();
        private final List<Long> placementRequestIds =
                new CopyOnWriteArrayList<>();

        private long placementStarted(BalanceContext context) {
            long requestId = context.getRequestId();
            totalAttempts.incrementAndGet();
            placementRequestIds.add(requestId);
            attemptsByRequest.computeIfAbsent(
                    requestId, ignored -> new AtomicInteger())
                    .incrementAndGet();
            BalanceContext original = contexts.putIfAbsent(requestId, context);
            if (original != null && original != context) {
                throw new AssertionError(
                        "placement replaced context for request " + requestId);
            }
            SchedulingMetadata metadata = context.schedulingMetadata();
            if (metadata != null) {
                long deadline = metadata.expiresAtMs();
                Long originalDeadline = deadlines.putIfAbsent(
                        requestId, deadline);
                if (originalDeadline != null
                        && originalDeadline != deadline) {
                    throw new AssertionError(
                            "placement changed deadline for request "
                                    + requestId);
                }
            }
            return System.nanoTime();
        }

        private void placementFinished(long startedAt) {
            placementLatenciesNanos.add(System.nanoTime() - startedAt);
        }

        private void recordCandidateEvaluation() {
            candidateEvaluations.incrementAndGet();
        }

        private void recordPlacementWakeup(PlacementKey key) {
            if (key.role() == RoleType.DECODE
                    && "capacity-contract".equals(key.group())) {
                placementWakeups.incrementAndGet();
            }
        }

        private int totalPlacementAttempts() {
            return totalAttempts.get();
        }

        private long candidateEvaluations() {
            return candidateEvaluations.get();
        }

        private long placementWakeups() {
            return placementWakeups.get();
        }

        private void resetPlacementWakeups() {
            placementWakeups.set(0L);
        }

        private BalanceContext context(long requestId) {
            return contexts.get(requestId);
        }

        private long deadline(long requestId) {
            return deadlines.getOrDefault(requestId, -1L);
        }

        private int maxAttemptsForAnyRequest() {
            return attemptsByRequest.values().stream()
                    .mapToInt(AtomicInteger::get)
                    .max()
                    .orElse(0);
        }

        private List<Long> requestIdsFrom(int attemptIndex) {
            return List.copyOf(placementRequestIds.subList(
                    Math.min(attemptIndex, placementRequestIds.size()),
                    placementRequestIds.size()));
        }

        private void print(
                String scenario,
                int requests,
                int pendingDepth,
                int statusTriggeredPlacements,
                long capacityReleasedAt) {
            long[] latencies = placementLatenciesNanos.stream()
                    .mapToLong(Long::longValue)
                    .sorted()
                    .toArray();
            long p50 = percentile(latencies, 0.50);
            long p99 = percentile(latencies, 0.99);
            long recoveryLatency = Math.max(
                    0L, System.nanoTime() - capacityReleasedAt);
            System.out.printf(
                    "FlexLB placement contract: scenario=%s requests=%d "
                            + "placement_attempts=%d attempts_per_request=%.4f "
                            + "max_attempts_per_request=%d placement_wakeups=%d "
                            + "candidate_evaluations=%d pending_depth=%d "
                            + "placement_latency_p50_us=%d "
                            + "placement_latency_p99_us=%d "
                            + "capacity_recovery_latency_us=%d "
                            + "status_triggered_placements=%d%n",
                    scenario,
                    requests,
                    totalPlacementAttempts(),
                    (double) totalPlacementAttempts() / requests,
                    maxAttemptsForAnyRequest(),
                    placementWakeups(),
                    candidateEvaluations(),
                    pendingDepth,
                    TimeUnit.NANOSECONDS.toMicros(p50),
                    TimeUnit.NANOSECONDS.toMicros(p99),
                    TimeUnit.NANOSECONDS.toMicros(recoveryLatency),
                    statusTriggeredPlacements);
        }

        private static long percentile(long[] sorted, double quantile) {
            if (sorted.length == 0) {
                return 0L;
            }
            int index = (int) Math.ceil(quantile * sorted.length) - 1;
            return sorted[Math.max(0, Math.min(index, sorted.length - 1))];
        }
    }

    private static final class QuietPlacementLogs implements AutoCloseable {
        private final Logger strategyLogger = (Logger) LoggerFactory.getLogger(
                RandomStrategy.class);
        private final Logger flexlbLogger = (Logger) LoggerFactory.getLogger(
                "flexlbLogger");
        private final Logger syncLogger = (Logger) LoggerFactory.getLogger(
                "syncLogger");
        private final Level previousStrategyLevel = strategyLogger.getLevel();
        private final Level previousFlexlbLevel = flexlbLogger.getLevel();
        private final Level previousSyncLevel = syncLogger.getLevel();

        private QuietPlacementLogs() {
            strategyLogger.setLevel(Level.ERROR);
            flexlbLogger.setLevel(Level.ERROR);
            syncLogger.setLevel(Level.ERROR);
        }

        @Override
        public void close() {
            strategyLogger.setLevel(previousStrategyLevel);
            flexlbLogger.setLevel(previousFlexlbLevel);
            syncLogger.setLevel(previousSyncLevel);
        }
    }

    private static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        config.queueScheduler().setOrdering(QueueOrderingConfig.priority());
        config.fixedWindowDecision().setMaxRequests(1);
        config.fixedWindowDecision().setMaxCollectionWaitMs(1L);
        config.queueScheduler().getCapacity()
                .setMaxOutstandingRequestsGlobal(64);
        config.queueScheduler().getCapacity()
                .setMaxWaitingRequestsPerPrefillWorker(8);
        ((DispatcherConfig) config.getDispatcher())
                .setMaxInflightBatchesPerPrefillWorker(2);
        config.getRouter().getRoles().getPrefill()
                .setSelector(new RoutingConfig.RandomPrefillSelectorConfig());
        config.getRouter().getRoles().getDecode()
                .setSelector(new RoutingConfig.RandomDecodeSelectorConfig());
        config.getRouter().getRoles().getPrefill().getAvailability()
                .setMaxPendingRequests(1L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxEngineRequests(1L);
        config.getRouter().getRoles().getDecode().getAvailability()
                .setMaxKvUsagePercent(100L);
        return config;
    }

    private static ModelMetaConfig modelMeta(boolean prefillFirst) {
        ModelMetaConfig meta = mock(
                ModelMetaConfig.class, withSettings().stubOnly());
        when(meta.requiredRoles()).thenReturn(
                prefillFirst
                        ? List.of(RoleType.PREFILL, RoleType.DECODE)
                        : List.of(RoleType.DECODE, RoleType.PREFILL));
        return meta;
    }

    private static WorkerStatus initializedStatus(
            RoleType role, String ip, int port) {
        WorkerStatus status = WorkerStatus.createDiscovered(
                role, "capacity-contract", ip, port, port + 1, null);
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
            RoleType role, long version, boolean saturated) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(role);
        response.setAlive(true);
        response.setStatusVersion(version);
        response.setLatestFinishedVersion(0L);
        response.setAvailableKvCacheTokens(1_000_000L);
        response.setTotalKvCacheTokens(2_000_000L);
        response.setMaxSeqLen(1_000_000L);
        response.setMaxBatchTokensSize(1_000_000L);
        if (saturated) {
            TaskInfo task = new TaskInfo();
            task.setRequestId(Fixture.EXTERNAL_REQUEST_ID);
            task.setPhase(role == RoleType.PREFILL
                    ? TaskPhase.PENDING : TaskPhase.RUNNING);
            task.setInputLength(128L);
            response.setRunningTaskInfo(Map.of(
                    Long.toString(Fixture.EXTERNAL_REQUEST_ID), task));
            response.setRunningQueryLen(role == RoleType.DECODE ? 1L : 0L);
            response.setWaitingQueryLen(role == RoleType.PREFILL ? 1L : 0L);
        } else {
            response.setRunningTaskInfo(Map.of());
        }
        response.setFinishedTaskInfo(Map.of());
        return response;
    }

    private static WorkerStatusResponse decodeRunningStatus(
            long version, int runningCount) {
        WorkerStatusResponse response =
                statusResponse(RoleType.DECODE, version, false);
        Map<String, TaskInfo> running = new LinkedHashMap<>();
        for (int index = 0; index < runningCount; index++) {
            long requestId = Fixture.EXTERNAL_REQUEST_ID + index;
            TaskInfo task = new TaskInfo();
            task.setRequestId(requestId);
            task.setPhase(TaskPhase.RUNNING);
            task.setInputLength(128L);
            running.put(Long.toString(requestId), task);
        }
        response.setRunningTaskInfo(running);
        response.setRunningQueryLen((long) runningCount);
        return response;
    }

    private static final class RecordingSubmissionPort
            implements BatchSubmissionPort {
        private final List<Command> commands = new CopyOnWriteArrayList<>();
        private final Semaphore commandSignals = new Semaphore(0);
        private final Semaphore preparationSignals = new Semaphore(0);
        private final Semaphore preparationReleases = new Semaphore(0);
        private final AtomicBoolean holdCompletions = new AtomicBoolean();
        private final AtomicBoolean blockPreparation = new AtomicBoolean();

        @Override
        public CapacityBoundary.Attempt<PreparedSubmission>
                tryPrepareSubmission() {
            if (blockPreparation.get()) {
                preparationSignals.release();
                preparationReleases.acquireUninterruptibly();
            }
            return CapacityBoundary.Attempt.accepted(
                    new PreparedSubmission() {
                        private boolean submitted;

                        @Override
                        public void submitBatch(
                                Command command,
                                BiConsumer<ScheduledRequest,
                                        SlotDeliveryPort.Completion> observer) {
                            if (submitted) {
                                throw new IllegalStateException(
                                        "prepared submission reused");
                            }
                            submitted = true;
                            commands.add(command);
                            commandSignals.release();
                            if (!holdCompletions.get()) {
                                for (ScheduledRequest item : command.exactItems()) {
                                    observer.accept(
                                            item,
                                            SlotDeliveryPort.Completion
                                                    .delivered());
                                }
                            }
                        }

                        @Override
                        public void close() {
                        }
                    });
        }

        private List<Long> requestIds() {
            return commands.stream()
                    .flatMap(command -> command.exactItems().stream())
                    .map(ScheduledRequest::requestId)
                    .toList();
        }

        private List<String> decodeAddresses() {
            return commands.stream()
                    .flatMap(command -> command.exactItems().stream())
                    .map(ScheduledRequest.class::cast)
                    .map(item -> item.decode().getServerIp() + ":"
                            + item.decode().getHttpPort())
                    .toList();
        }

        private void blockPreparation() {
            blockPreparation.set(true);
        }

        private boolean awaitPreparation(long timeout, TimeUnit unit)
                throws InterruptedException {
            return preparationSignals.tryAcquire(timeout, unit);
        }

        private void releasePreparation() {
            preparationReleases.release();
        }

        private void holdCompletions() {
            holdCompletions.set(true);
        }

        private boolean awaitCommands(
                int count, long timeout, TimeUnit unit)
                throws InterruptedException {
            return commandSignals.tryAcquire(count, timeout, unit);
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
