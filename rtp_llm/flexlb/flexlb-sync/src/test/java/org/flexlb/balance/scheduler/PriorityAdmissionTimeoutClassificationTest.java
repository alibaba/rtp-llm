package org.flexlb.balance.scheduler;

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

import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** End-to-end coverage for the existing typed PRIORITY admission-timeout contract. */
class PriorityAdmissionTimeoutClassificationTest {

    private static final String PREFILL_IP_PORT = "10.0.0.1:8080";

    private PriorityScheduler scheduler;
    private EndpointRegistry endpointRegistry;
    private PrefillEndpoint prefillEndpoint;
    private FlexlbConfig config;

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
        ConfigService configService = mock(ConfigService.class);
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
    }

    @AfterEach
    void tearDown() {
        scheduler.shutdown();
    }

    @Test
    void queuedTimeoutReportsHigherPriorityAhead() {
        long now = System.currentTimeMillis();
        BatchItem blocker = priorityItem(1L, 70, now);
        BatchItem victim = priorityItem(2L, 50, now + 1);
        enqueue(blocker);
        enqueue(victim);

        scheduler.onTimeout(victim, new TimeoutException("priority admission timed out"));

        assertAdmissionFailure(victim, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.HIGHER_PRIORITY_AHEAD);
        assertEquals(List.of(1L), queuedRequestIds());
    }

    @Test
    void queuedTimeoutReportsEarlierSamePriorityAhead() {
        long now = System.currentTimeMillis();
        BatchItem blocker = priorityItem(11L, 50, now);
        BatchItem victim = priorityItem(12L, 50, now + 1);
        enqueue(blocker);
        enqueue(victim);

        scheduler.onTimeout(victim, new TimeoutException("priority admission timed out"));

        assertAdmissionFailure(victim, StrategyErrorType.PRIORITY_ADMISSION_REJECTED,
                AdmissionRejectReason.SAME_PRIORITY_AHEAD);
        assertEquals(List.of(11L), queuedRequestIds());
    }

    @Test
    void queuedHeadTimeoutReportsResourceExhausted() {
        long now = System.currentTimeMillis();
        BatchItem victim = priorityItem(21L, 70, now);
        BatchItem lowerBehind = priorityItem(22L, 30, now + 1);
        enqueue(victim);
        enqueue(lowerBehind);

        scheduler.onTimeout(victim, new TimeoutException("priority admission timed out"));

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertEquals(List.of(22L), queuedRequestIds());
    }

    @Test
    void enqueueBatchTimeoutWithoutQueueEvidenceReportsResourceExhausted() {
        BatchItem victim = priorityItem(31L, 50, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(victim));

        scheduler.onTimeout(victim, new TimeoutException("EnqueueBatch deadline"));

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertTrue(queuedRequestIds().isEmpty());
    }

    @Test
    void inflightTtlWithoutQueueEvidenceReportsResourceExhausted() {
        BatchItem victim = priorityItem(41L, 50, System.currentTimeMillis());
        assertTrue(scheduler.registerInflight(victim));
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(-1);

        scheduler.cleanupInflight();

        assertAdmissionFailure(victim, StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        assertTrue(queuedRequestIds().isEmpty());
    }

    @Test
    void fifoRegistrationKeepsBatchSloExpired() {
        SchedulingTestConfig.useFifoQueue(config);
        BalanceContext context = context(51L, 50);
        long now = System.currentTimeMillis();
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 3_600_000));
        CompletableFuture<Response> future = scheduler.submit(context);
        config.queueScheduler().getLifecycle().setStaleInflightTimeoutMs(-1);

        scheduler.cleanupInflight();

        Response response = future.join();
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.UNSPECIFIED, response.getAdmissionRejectReason());
    }

    @RepeatedTest(20)
    @Timeout(5)
    void fifoNonBatchQueuedDeadlineUsesSharedSchedulingTimeoutCode() throws Exception {
        restartFifoNonBatchWithOneInflight();
        assertTrue(prefillEndpoint.tryCommitRequest(9_999L, 1_000L, 1));
        BalanceContext context = context(52L, 50);
        context.setConfig(config);
        long now = System.currentTimeMillis();
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 3_600_000));
        CompletableFuture<Response> future = scheduler.submit(context);
        long deadline = System.currentTimeMillis() + 1_000;
        while (prefillEndpoint.getBatcher().queueSize() != 1
                && System.currentTimeMillis() < deadline) {
            Thread.onSpinWait();
        }
        assertEquals(1, prefillEndpoint.getBatcher().queueSize());

        scheduler.onRequestExpired(52L, future);

        Response response = future.get(1, TimeUnit.SECONDS);
        assertFalse(response.isSuccess());
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), response.getCode());
        assertEquals(AdmissionRejectReason.UNSPECIFIED, response.getAdmissionRejectReason());
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
        assertTrue(prefillEndpoint.getBatcher().tryOffer(item));
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
