package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.BatchSubmissionPort;
import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.SlotDeliveryPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.resource.DecodeResourceMeasure;
import org.flexlb.balance.resource.PrefillResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.FifoOrderingConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
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
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.BiConsumer;
import java.util.stream.LongStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

/** External contract for temporary P/D capacity pressure. */
class TransientCapacityQueueContractTest {

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
        config.queueScheduler().setOrdering(new FifoOrderingConfig());
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
        config.queueScheduler().setOrdering(new FifoOrderingConfig());
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
        config.queueScheduler().setOrdering(new FifoOrderingConfig());
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
        config.setDispatcher(new NonBatchDispatcherConfig());
        verifyPoolCapacityWake(config, 602L);
    }

    private static void verifyPoolCapacityWake(
            FlexlbConfig config,
            long requestId) throws Exception {
        config.queueScheduler().setOrdering(new FifoOrderingConfig());
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
        private final RequestSchedulerTestRuntime runtime;
        private final WorkerStatus prefillStatus;
        private final WorkerStatus decodeStatus;
        private final DecodeEndpoint decodeEndpoint;
        private final DecodeResourceMeasure decodeMeasure;
        private final RoleType saturatedRole;

        private Fixture(RoleType saturatedRole) {
            this(saturatedRole, config());
        }

        private Fixture(RoleType saturatedRole, FlexlbConfig config) {
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
            runtime.endpointRegistry().registerPreinitializedEndpoint(
                            RoleType.PREFILL,
                            PREFILL_ADDRESS,
                            prefillStatus);
            decodeEndpoint = (DecodeEndpoint) runtime.endpointRegistry()
                    .registerPreinitializedEndpoint(
                    RoleType.DECODE,
                    DECODE_ADDRESS,
                    decodeStatus);

            if (saturatedRole == RoleType.PREFILL) {
                runtime.applyStatus(
                        prefillStatus,
                        statusResponse(RoleType.PREFILL, 2L, true));
            } else if (saturatedRole == RoleType.DECODE) {
                runtime.applyStatus(
                        decodeStatus,
                        statusResponse(RoleType.DECODE, 2L, true));
            }

            EngineWorkerStatus workers =
                    new EngineWorkerStatus(runtime.endpointRegistry());
            decodeMeasure = new DecodeResourceMeasure(configService);
            ResourceMeasureFactory measures = new ResourceMeasureFactory(
                    List.of(
                            new PrefillResourceMeasure(configService),
                            decodeMeasure));
            ConfiguredLoadBalanceSelector selector =
                    new ConfiguredLoadBalanceSelector(
                            List.of(
                                    new RandomStrategy(workers, measures),
                                    new CostBasedDecodeStrategy(workers, measures)));
            runtime.bindRouter(new DefaultRouter(
                    selector,
                    ignored -> GroupRoutingDecision.none(),
                    modelMeta(),
                    runtime.placementAvailability()));
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

    private static FlexlbConfig config() {
        FlexlbConfig config = new FlexlbConfig();
        config.queueScheduler().setOrdering(new PriorityOrderingConfig());
        config.fixedWindowDecision().setMaxRequests(1);
        config.fixedWindowDecision().setMaxCollectionWaitMs(1L);
        config.queueScheduler().getCapacity()
                .setMaxOutstandingRequestsGlobal(64);
        config.queueScheduler().getCapacity()
                .setMaxWaitingRequestsPerPrefillWorker(8);
        ((BatchDispatcherConfig) config.getDispatcher())
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

    private static ModelMetaConfig modelMeta() {
        ModelMetaConfig meta = mock(
                ModelMetaConfig.class, withSettings().stubOnly());
        when(meta.requiredRoles()).thenReturn(
                List.of(RoleType.DECODE, RoleType.PREFILL));
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
            return new CapacityBoundary.Attempt.Accepted<>(
                    new PreparedSubmission() {
                        private boolean submitted;

                        @Override
                        public void submitBatch(
                                Command command,
                                BiConsumer<DeliveryItem,
                                        SlotDeliveryPort.Completion> observer) {
                            if (submitted) {
                                throw new IllegalStateException(
                                        "prepared submission reused");
                            }
                            submitted = true;
                            commands.add(command);
                            commandSignals.release();
                            if (!holdCompletions.get()) {
                                for (DeliveryItem item : command.exactItems()) {
                                    observer.accept(
                                            item,
                                            SlotDeliveryPort.Completion
                                                    .Delivered.INSTANCE);
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
                    .map(DeliveryItem::requestId)
                    .toList();
        }

        private List<String> decodeAddresses() {
            return commands.stream()
                    .flatMap(command -> command.exactItems().stream())
                    .map(BatchItem.class::cast)
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
