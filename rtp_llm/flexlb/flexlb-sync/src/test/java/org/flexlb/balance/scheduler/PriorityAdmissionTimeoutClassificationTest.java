package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.priority.UnsupportedEngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.RepeatedTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Queue expiration reads the last scheduling decision; it does not reclassify queue occupants. */
class PriorityAdmissionTimeoutClassificationTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";

    private PriorityScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private PrefillEndpoint prefillEndpoint;
    private FlexlbConfig config;
    private ConfigService configService;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxCollectionWaitMs(3_600_000);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxRequests(100);
        SchedulingTestConfig.useBatchDispatcher(config).setMaxWaitingRequestsPerPrefillWorker(100);
        startScheduler();
    }

    private void startScheduler() {
        startScheduler(RouteDecisionDelivery.INSTANCE,
                PriorityScheduler.CompletionExecutorPolicy.productionDefaults());
    }

    private void startScheduler(
            DecisionDelivery<List<BatchItem>> routeDecisionDelivery,
            PriorityScheduler.CompletionExecutorPolicy completionExecutorPolicy) {
        configService = mock(ConfigService.class);
        Router router = mock(Router.class);
        BatchDispatcher dispatcher = mock(BatchDispatcher.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);

        when(configService.loadBalanceConfig()).thenReturn(config);
        when(router.route(any(BalanceContext.class))).thenAnswer(invocation ->
                route(invocation.<BalanceContext>getArgument(0).getRequestId()));

        endpointRegistry = new EndpointRegistry(configService, () -> scheduler, reporter);
        scheduler = new PriorityScheduler(configService, router, endpointRegistry,
                dispatcher, reporter, null, null,
                new UnsupportedEngineCancelChannel(),
                PriorityScheduler.EngineFencePolicy.productionDefaults(),
                routeDecisionDelivery, completionExecutorPolicy);

        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        endpointRegistry.ensureEndpoint(RoleType.PREFILL, PREFILL_IP_PORT, status);
        prefillEndpoint = endpointRegistry.getPrefill(PREFILL_IP_PORT);
        // Two 128-token requests cannot share a batch at this strict padded-token limit.
        status.setMaxBatchTokensSize(256);
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @ParameterizedTest
    @CsvSource({"70", "50", "0"})
    void queueExpirationReadsWaitStateRegardlessOfOccupantPriority(int blockerPriority) {
        long now = System.currentTimeMillis();
        enqueue(priorityItem(1L, blockerPriority, now));
        BatchItem item = priorityItem(2L, 50, now + 1);
        enqueue(item);
        awaitWaitReason("batch collection window exceeded request scheduling budget");

        scheduler.onRequestExpired(item.requestId(), item.future());

        assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertTrue(item.future().join().getErrorMessage().contains("batch collection window"));
        assertEquals(List.of(1L), queuedRequestIds());
    }

    @Test
    void queuedHeadTimeoutReportsResourceExhausted() {
        long now = System.currentTimeMillis();
        BatchItem victim = priorityItem(21L, 70, now);
        BatchItem lowerBehind = priorityItem(22L, 30, now + 1);
        config.batchDispatcher().setMaxInflightBatchesPerPrefillWorker(1);
        prefillEndpoint.commitBatch(90_001L, 1_000L, List.of(priorityItem(90_002L, 30, now)));
        enqueue(victim);
        enqueue(lowerBehind);

        scheduler.onTimeout(victim, new TimeoutException("priority admission timed out"));

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertEquals(List.of(22L), queuedRequestIds());
    }

    @Test
    void lowReportedKvAloneDoesNotProveAQueuedHeadIsCapacityBlocked() {
        BatchItem item = priorityItem(90_010L, 50, System.currentTimeMillis());
        prefillEndpoint.getStatus().getTotalKvCacheTokens().set(128);
        prefillEndpoint.getStatus().getAvailableKvCacheTokens().set(0);
        enqueue(item);

        scheduler.onRequestExpired(item.requestId(), item.future());

        assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @ParameterizedTest
    @CsvSource({"70", "50"})
    void sameBatchMemberIsNotAPriorityBlocker(int priority) {
        prefillEndpoint.getStatus().setMaxBatchTokensSize(1_024);
        long now = System.currentTimeMillis();
        enqueue(priorityItem(90_030L, priority, now));
        BatchItem item = priorityItem(90_031L, 50, now + 1);
        enqueue(item);

        scheduler.onRequestExpired(item.requestId(), item.future());

        assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertEquals(List.of(90_030L), queuedRequestIds());
    }

    @Test
    void batchGateUsesWorkerConfigurationNotRequestConfiguration() {
        BatchItem item = priorityItem(90_040L, 50, System.currentTimeMillis());
        SchedulingTestConfig.useBatchDispatcher(item.ctx().getConfig()).setMaxInflightBatchesPerPrefillWorker(1);
        prefillEndpoint.commitBatch(90_041L, 1_000L, List.of(priorityItem(90_042L, 30, System.currentTimeMillis())));
        enqueue(item);

        scheduler.onRequestExpired(item.requestId(), item.future());

        // The worker's actual gate is unlimited; the request snapshot cannot create a gate.
        assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void decodeGateUsesLiveConfigurationAfterLimitIsRelaxed() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.2");
        status.setPort(8080);
        DecodeEndpoint decode = new DecodeEndpoint(status);
        decode.reserve(90_050L, 0, 0, 50);
        decode.markQueuedPhase(90_050L);
        decode.reserve(90_051L, 0, 0, 70);
        BalanceContext context = context(90_050L, 50);
        context.getConfig().getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        FlexlbConfig updated = new FlexlbConfig();
        updated.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(2L);
        when(configService.loadBalanceConfig()).thenReturn(updated);
        Response route = route(90_050L);
        BatchItem item = new BatchItem(context, new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL), null,
                prefillEndpoint, decode, System.currentTimeMillis());
        enqueue(item);

        scheduler.onRequestExpired(item.requestId(), item.future());

        assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @ParameterizedTest
    @CsvSource({"70", "50", "0", "101", "30"})
    void decodeEngineSlotGateReportsCapacityRegardlessOfOccupantPriority(int blockerPriority) {
        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.2");
        status.setPort(8080);
        status.getTotalKvCacheTokens().set(128);
        status.getAvailableKvCacheTokens().set(128);
        DecodeEndpoint decode = org.mockito.Mockito.spy(new DecodeEndpoint(status));
        // The queued request already owns all available KV. Only the engine
        // slot can stop this delivery; recharging its prompt would invent OOM.
        decode.reserve(90_020L, 128, 128, 50);
        decode.markQueuedPhase(90_020L);
        decode.reserve(90_021L, 0, 0, blockerPriority);
        BalanceContext context = context(90_020L, 50);
        config.getRouter().getRoles().getDecode().getAvailability().setMaxEngineRequests(1L);
        config.batchDispatcher().setMaxCollectionWaitMs(0);
        Response route = route(90_020L);
        BatchItem item = new BatchItem(context, new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL), null,
                prefillEndpoint, decode, System.currentTimeMillis());
        enqueue(item);

        awaitWaitReason("decode engine slots exhausted");

        scheduler.onRequestExpired(item.requestId(), item.future());

        assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertEquals(1, decode.getEngineLoad());
        assertTrue(item.future().join().getErrorMessage().contains("decode engine slots exhausted"));
        // This isolated Decode endpoint is not registered for router rollback.
        decode.release(item.requestId());
        java.util.Map<String, Object> diagnostics = context.getSchedulingDiagnostics();
        assertTrue(diagnostics.get("cause").toString().contains("decode engine slots exhausted"));
        java.util.Map<?, ?> decodeSnapshot = (java.util.Map<?, ?>) ((java.util.List<?>) diagnostics.get("decode")).getFirst();
        assertEquals(2, decodeSnapshot.get("reservedRequests"), "capture before timeout releases KV");
        assertEquals(128L, decodeSnapshot.get("hardKvReserved"));
        assertEquals(0L, decodeSnapshot.get("kvAvailable"));
        assertEquals(1, decode.getInflightCount());
        assertEquals(0L, decode.inflightHardKvReserved());
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.never()).layeredAdmissionView();
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.never()).getAcceptedLayerCount();
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.never()).getRunningLayerCount();
    }

    @Test
    void decodeDispatchRetriesDoNotReadOccupancySnapshotsAndClearQueueWaitOnSuccess() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("10.0.0.2");
        status.setPort(8080);
        DecodeEndpoint decode = org.mockito.Mockito.spy(new DecodeEndpoint(status));
        decode.reserve(90_100L, 0, 0, 50);
        decode.markQueuedPhase(90_100L);
        decode.reserve(90_101L, 0, 0, 70);
        Response response = route(90_100L);
        BatchItem item = new BatchItem(context(90_100L, 50), new CompletableFuture<>(), response,
                PriorityScheduler.findServer(response, RoleType.PREFILL), null,
                prefillEndpoint, decode, System.currentTimeMillis());

        assertEquals(DecodeEndpoint.DispatchClaimResult.CAPACITY_FULL,
                PriorityScheduler.tryClaimDecodeDispatch(item, 1));
        assertEquals("decode engine slots exhausted", prefillEndpoint.getBatcher().getWaitReason());
        for (int attempt = 0; attempt < 100; attempt++) {
            assertEquals(DecodeEndpoint.DispatchClaimResult.CAPACITY_FULL,
                    PriorityScheduler.tryClaimDecodeDispatch(item, 1));
        }
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.never()).layeredAdmissionView();
        org.mockito.Mockito.verify(decode, org.mockito.Mockito.never()).admissionVersion();

        decode.release(90_101L);
        assertEquals(DecodeEndpoint.DispatchClaimResult.CLAIMED,
                PriorityScheduler.tryClaimDecodeDispatch(item, 1));
        assertNull(prefillEndpoint.getBatcher().getWaitReason());

        prefillEndpoint.getBatcher().setWaitReason("decode engine slots exhausted");
        decode.release(90_100L);
        assertEquals(DecodeEndpoint.DispatchClaimResult.NOT_OWNED,
                PriorityScheduler.tryClaimDecodeDispatch(item, 1));
        assertNull(prefillEndpoint.getBatcher().getWaitReason(), "lost ownership must not look like a retryable capacity rejection");
    }

    @ParameterizedTest
    @CsvSource({"prefill request slots exhausted", "decode engine slots exhausted"})
    void expirationBeforeFirstLoopDecisionReadsExistingQueueState(String reason) {
        BatchItem item = priorityItem(90_060L, 50, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(item));
        WorkerBatcher batcher = prefillEndpoint.getBatcher();
        batcher.setWaitReason("batch collection window exceeded request scheduling budget");
        batcher.setWaitReason(reason);
        when(configService.loadBalanceConfig()).thenThrow(new AssertionError("expiration must not reclassify"));
        try {
            scheduler.onRequestExpired(item.requestId(), item.future());
            assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED);
            assertTrue(item.future().join().getErrorMessage().contains(reason + "; context="));
            assertEquals(reason, batcher.getWaitReason());
        } finally {
            org.mockito.Mockito.doReturn(config).when(configService).loadBalanceConfig();
        }
    }

    @Test
    void unobservedPrefillCapacityDoesNotInventQueueWaitReason() {
        config.batchDispatcher().setMaxInflightBatchesPerPrefillWorker(1);
        long now = System.currentTimeMillis();
        prefillEndpoint.commitBatch(90_070L, 1000, List.of(priorityItem(90_071L, 70, now)));
        BatchItem item = priorityItem(90_072L, 50, now);
        assertTrue(scheduler.registerInflight(item));

        scheduler.onRequestExpired(item.requestId(), item.future());

        assertAdmissionFailure(item, StrategyErrorType.RESOURCE_EXHAUSTED, AdmissionRejectReason.RESOURCE_EXHAUSTED);
    }

    @Test
    void successfulDeliveryClaimClearsEarlierCapacityWait() throws Exception {
        scheduler.shutdown();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightRequestsPerPrefillWorker(1);
        ControlledRouteDelivery delivery = new ControlledRouteDelivery();
        startScheduler(delivery, PriorityScheduler.CompletionExecutorPolicy.productionDefaults());
        assertTrue(prefillEndpoint.tryCommitRequest(90_080L, 1000, 1));
        WorkerBatcher batcher = prefillEndpoint.getBatcher();
        long version = batcher.queueVersion();
        CompletableFuture<Response> future = scheduler.submit(expiringContext(90_081L));
        awaitCapacityBlockedReadyState(batcher, version);

        assertTrue(prefillEndpoint.releaseRequest(90_080L));
        PendingRoute delivered = delivery.awaitNext();
        assertNull(batcher.getWaitReason());
        delivered.succeed();
        assertTrue(future.get(1, TimeUnit.SECONDS).isSuccess());
    }

    @Test
    void registeredRequestAwaitingMasterSchedulingReportsAdmissionCapacity() {
        BatchItem victim = priorityItem(31L, 50, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(victim));

        scheduler.onTimeout(victim, new TimeoutException("EnqueueBatch deadline"));

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertTrue(queuedRequestIds().isEmpty());
    }

    @Test
    void unclaimedInflightTtlReportsMasterSchedulingCapacity() {
        BatchItem victim = priorityItem(41L, 50, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(victim));
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(-1);

        scheduler.cleanupInflight();

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertTrue(queuedRequestIds().isEmpty());
    }

    @Test
    void fifoCollectionWindowExceedingBudgetReportsAdmissionCapacity() {
        SchedulingTestConfig.useFifoQueue(config);
        BalanceContext context = context(51L, 50);
        long now = System.currentTimeMillis();
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 3_600_000));
        CompletableFuture<Response> future = scheduler.submit(context);
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(-1);

        scheduler.cleanupInflight();

        Response response = future.join();
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED, response.getAdmissionRejectReason());
    }

    @RepeatedTest(20)
    @Timeout(5)
    void fifoNonBatchQueuedDeadlineReportsAdmissionCapacityExhausted() throws Exception {
        restartFifoNonBatchWithOneInflight();
        assertTrue(prefillEndpoint.tryCommitRequest(9_999L, 1_000L, 1));
        WorkerBatcher batcher = prefillEndpoint.getBatcher();
        long beforeSubmitVersion = batcher.queueVersion();
        BalanceContext context = context(52L, 50);
        context.setConfig(config);
        long now = System.currentTimeMillis();
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 3_600_000));
        CompletableFuture<Response> future = scheduler.submit(context);
        awaitCapacityBlockedReadyState(batcher, beforeSubmitVersion);

        scheduler.onRequestExpired(52L, future);

        Response response = future.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.RESOURCE_EXHAUSTED, response.getAdmissionRejectReason());
        assertTrue(response.getErrorMessage().contains("prefill request slots exhausted"));
    }

    @Test
    @Timeout(5)
    void fifoNonBatchCapacityReleaseWakesReadyRequestWithoutDeadlineOrNewRequest()
            throws Exception {
        restartFifoNonBatchWithOneInflight();
        assertTrue(prefillEndpoint.tryCommitRequest(9_997L, 1_000L, 1));
        WorkerBatcher batcher = prefillEndpoint.getBatcher();
        long beforeSubmitVersion = batcher.queueVersion();

        CompletableFuture<Response> future = scheduler.submit(expiringContext(57L));
        awaitCapacityBlockedReadyState(batcher, beforeSubmitVersion);
        assertFalse(future.isDone());

        assertTrue(prefillEndpoint.releaseRequest(9_997L));

        assertTrue(future.get(1, TimeUnit.SECONDS).isSuccess());
        assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                scheduler.getRequestState(57L, 0).state());
        awaitQueueSize(0);
    }

    @Test
    @Timeout(5)
    void fifoNonBatchQueuedClientCancelCompletesOwnedPublication() throws Exception {
        restartFifoNonBatchWithOneInflight();
        assertTrue(prefillEndpoint.tryCommitRequest(9_998L, 1_000L, 1));
        BalanceContext context = context(53L, 50);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 3_600_000));
        CompletableFuture<Response> future = scheduler.submit(context);
        awaitQueueSize(1);

        RequestLifecycleSnapshot cancelled = scheduler.cancelRequest(
                53L, 0, CancelReason.CLIENT_CANCELLED);

        assertEquals(RequestLifecycleState.CANCELLED, cancelled.state());
        Response response = future.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.REQUEST_CANCELLED.getErrorCode(), response.getCode());
    }

    @Test
    @Timeout(5)
    void claimedNonBatchSuccessSurvivesLaterDeadlineBeforeAsyncPublication()
            throws Exception {
        scheduler.shutdown();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(10);
        ControlledRouteDelivery delivery = new ControlledRouteDelivery();
        startScheduler(delivery, new PriorityScheduler.CompletionExecutorPolicy(1, 2));

        CompletableFuture<Response> blockerFuture = scheduler.submit(
                expiringContext(54L));
        PendingRoute blocker = delivery.awaitNext();
        CountDownLatch completionWorkerBlocked = new CountDownLatch(1);
        CountDownLatch releaseCompletionWorker = new CountDownLatch(1);
        blockerFuture.thenRun(() -> {
            completionWorkerBlocked.countDown();
            await(releaseCompletionWorker);
        });
        blocker.succeed();
        assertTrue(completionWorkerBlocked.await(1, TimeUnit.SECONDS));

        CompletableFuture<Response> successFuture = scheduler.submit(
                expiringContext(55L));
        PendingRoute success = delivery.awaitNext();
        try {
            success.succeed();
            awaitCompletionQueueSize(1);
            assertFalse(successFuture.isDone());

            scheduler.onRequestExpired(55L, successFuture);
            assertEquals(RequestLifecycleState.ACKNOWLEDGED,
                    scheduler.getRequestState(55L, 0).state());
        } finally {
            releaseCompletionWorker.countDown();
        }

        assertTrue(successFuture.get(1, TimeUnit.SECONDS).isSuccess());
    }

    private void restartFifoNonBatchWithOneInflight() {
        scheduler.shutdown();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useNonBatchDispatcher(config)
                .setMaxInflightRequestsPerPrefillWorker(1);
        startScheduler();
    }

    private BalanceContext expiringContext(long requestId) {
        BalanceContext context = context(requestId, 50);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 3_600_000));
        return context;
    }

    private void awaitQueueSize(int expected) {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
        while (prefillEndpoint.getBatcher().queueSize() != expected
                && System.nanoTime() < deadlineNanos) {
            Thread.onSpinWait();
        }
        assertEquals(expected, prefillEndpoint.getBatcher().queueSize());
    }

    private static void awaitCapacityBlockedReadyState(
            WorkerBatcher batcher, long beforeSubmitVersion) {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
        while (!(batcher.queueSize() == 1
                && batcher.queueVersion() >= beforeSubmitVersion + 2
                && batcher.isWaitingForSignal())
                && System.nanoTime() < deadlineNanos) {
            Thread.onSpinWait();
        }
        assertEquals(1, batcher.queueSize());
        assertTrue(batcher.queueVersion() >= beforeSubmitVersion + 2,
                "request did not transition from active FIFO work to ready delivery");
        assertTrue(batcher.isWaitingForSignal(),
                "capacity-blocked FIFO worker did not park on the shared condition");
    }

    private void awaitCompletionQueueSize(int expected) {
        long deadlineNanos = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
        while (scheduler.completionExecutorSnapshot().queueSize() != expected
                && System.nanoTime() < deadlineNanos) {
            Thread.onSpinWait();
        }
        assertEquals(expected, scheduler.completionExecutorSnapshot().queueSize());
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(2, TimeUnit.SECONDS));
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(interrupted);
        }
    }

    private void enqueue(BatchItem item) {
        assertTrue(scheduler.registerInflight(item));
        assertNull(prefillEndpoint.getBatcher().tryOffer(item));
    }

    private void awaitWaitReason(String expected) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        WorkerBatcher batcher = prefillEndpoint.getBatcher();
        while (!expected.equals(batcher.getWaitReason()) && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
        assertEquals(expected, batcher.getWaitReason());
    }

    private List<Long> queuedRequestIds() {
        return prefillEndpoint.getBatcher().queueManager().snapshot().items().stream()
                .map(snapshot -> snapshot.requestId())
                .toList();
    }

    private static void assertAdmissionFailure(BatchItem item,
                                               StrategyErrorType errorType,
                                               AdmissionRejectReason reason) {
        Response response = item.future().join();
        assertFalse(response.isSuccess());
        assertEquals(errorType.getErrorCode(), response.getCode());
        assertEquals(reason, response.getAdmissionRejectReason());
    }

    private BatchItem priorityItem(long requestId, int priority, long enqueuedAtMs) {
        BalanceContext context = context(requestId, priority);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority, enqueuedAtMs + 3_600_000));
        return item(context, enqueuedAtMs);
    }

    private BatchItem item(BalanceContext context, long enqueuedAtMs) {
        Response route = route(context.getRequestId());
        return new BatchItem(context, new CompletableFuture<>(), route,
                PriorityScheduler.findServer(route, RoleType.PREFILL), null,
                prefillEndpoint, null, enqueuedAtMs);
    }

    private static BalanceContext context(long requestId, int priority) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        request.setMaxNewTokens(8);
        request.setNumBeams(1);
        request.setModel("test-model");
        request.setPriority(priority);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(new FlexlbConfig());
        return context;
    }

    private static Response route(long requestId) {
        ServerStatus prefill = new ServerStatus();
        prefill.setSuccess(true);
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp("10.0.0.1");
        prefill.setHttpPort(8080);
        prefill.setGrpcPort(8081);
        prefill.setRequestId(requestId);
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(List.of(prefill));
        return response;
    }

    private static final class ControlledRouteDelivery
            implements DecisionDelivery<List<BatchItem>> {
        private final BlockingQueue<PendingRoute> pending = new LinkedBlockingQueue<>();

        @Override
        public void deliver(List<BatchItem> items, Callback callback) {
            if (items.size() != 1) {
                throw new IllegalArgumentException("expected one NON_BATCH route decision");
            }
            pending.add(new PendingRoute(items.getFirst(), callback));
        }

        private PendingRoute awaitNext() throws InterruptedException {
            PendingRoute route = pending.poll(1, TimeUnit.SECONDS);
            assertNotNull(route, "route decision was not delivered before timeout");
            return route;
        }
    }

    private record PendingRoute(BatchItem item, DecisionDelivery.Callback callback) {
        private void succeed() {
            callback.onDelivered(item);
        }
    }
}
