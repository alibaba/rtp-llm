package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DecisionGroupMetadata;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.flexlb.constant.MetricConstant.INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_REQUEST_COUNT;

class PrefillRequestLedgerTest {

    private PrefillEndpoint endpoint;
    private BatchSchedulerReporter reporter;
    private RecordingMonitor monitor;

    @BeforeEach
    void setUp() {
        WorkerStatus status = new WorkerStatus();
        status.setIp("127.0.0.41");
        status.setPort(8041);
        status.setGrpcPort(9041);
        status.setRole(RoleType.PREFILL);

        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchQueueMaxSize(128);
        config.setFlexlbBatchFixedWaitMs(60_000);
        config.setCostFormula("10 + 0.1*sum(computeTokens) + 5*batchSize");

        monitor = new RecordingMonitor();
        reporter = new BatchSchedulerReporter(monitor);
        endpoint = new PrefillEndpoint(status, config, noopHandler(), reporter);
    }

    @AfterEach
    void tearDown() {
        endpoint.close();
    }

    @Test
    void requestCommitAndReleaseAreIdempotentAndDoNotAffectBatchCount() {
        BatchItem request = batchItem(101L);

        assertTrue(endpoint.tryCommitRequest(request.requestId(), 700, 4));
        assertTrue(endpoint.tryCommitRequest(request.requestId(), 9_999, 4));
        assertEquals(1, endpoint.getInflightRequestCount());
        assertEquals(1, endpoint.getInflightRouteRequestCount());
        assertEquals(0, endpoint.getInflightBatchCount());
        assertEquals(1, endpoint.realPendingCount());

        assertTrue(endpoint.releaseRequest(101L));
        assertFalse(endpoint.releaseRequest(101L));
        assertEquals(0, endpoint.getInflightRequestCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());
    }

    @Test
    void routeRequestCapRemainsIndependentFromRealBatchMembers() {
        endpoint.commitBatch(900L, 1_000, List.of(batchItem(1L), batchItem(2L)));

        assertEquals(2, endpoint.availableRequestSlots(2));
        assertTrue(endpoint.tryCommitRequest(3L, 500, 2));
        assertTrue(endpoint.tryCommitRequest(4L, 500, 2));
        assertFalse(endpoint.tryCommitRequest(5L, 500, 2));
        assertEquals(4, endpoint.getInflightRequestCount());
        assertEquals(1, endpoint.getInflightBatchCount());
        assertEquals(2, endpoint.getInflightRouteRequestCount());

        endpoint.releaseRequest(3L);
        endpoint.releaseRequest(4L);
        endpoint.releaseBatch(900L);
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void laterBatchCommitCannotBreakRouteRequestHardCap() {
        assertTrue(endpoint.tryCommitRequest(11L, 500, 2));
        assertTrue(endpoint.tryCommitRequest(12L, 500, 2));

        endpoint.commitBatch(901L, 1_000,
                List.of(batchItem(21L), batchItem(22L), batchItem(23L)));

        assertEquals(2, endpoint.getInflightRouteRequestCount());
        assertEquals(5, endpoint.getInflightRequestCount());
        assertEquals(0, endpoint.availableRequestSlots(2));
        assertFalse(endpoint.tryCommitRequest(13L, 500, 2));
    }

    @Test
    void concurrentRequestCommitsNeverExceedHardCap() throws Exception {
        int attempts = 96;
        int cap = 11;
        ExecutorService executor = Executors.newFixedThreadPool(16);
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(attempts);
        AtomicInteger admitted = new AtomicInteger();
        try {
            for (int i = 0; i < attempts; i++) {
                long requestId = 10_000L + i;
                executor.execute(() -> {
                    await(start);
                    if (endpoint.tryCommitRequest(requestId, 100, cap)) {
                        admitted.incrementAndGet();
                    }
                    done.countDown();
                });
            }

            start.countDown();
            assertTrue(done.await(5, TimeUnit.SECONDS));

            assertEquals(cap, admitted.get());
            assertEquals(cap, endpoint.getInflightRequestCount());
            assertEquals(cap, endpoint.getInflightRouteRequestCount());
            assertEquals(0, endpoint.availableRequestSlots(cap));
        } finally {
            start.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void repeatedCommitReleaseCyclesLeaveNoAccountingBehind() {
        for (long requestId = 1; requestId <= 5_000; requestId++) {
            assertTrue(endpoint.tryCommitRequest(requestId, 10, 1));
            assertTrue(endpoint.releaseRequest(requestId));
        }

        assertEquals(0, endpoint.getInflightRequestCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());
        assertEquals(0, endpoint.realPendingCount());
    }

    @Test
    void workerStatusSettlesRequestByRequestIdRegardlessOfBatchId() {
        assertTrue(endpoint.tryCommitRequest(77L, 1_000, 4));

        TaskInfo finished = taskInfo(77L, 123_456L, null);
        updateWorkerStatus(Map.of("77", finished), Map.of());
        updateWorkerStatus(Map.of("77", finished), Map.of());

        assertEquals(0, endpoint.getInflightRequestCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());
    }

    @Test
    void runningObservationRefreshesRequestInactivityTtl() throws Exception {
        assertTrue(endpoint.tryCommitRequest(88L, 1_000, 4));
        Thread.sleep(25);

        TaskInfo running = taskInfo(88L, -1L, TaskPhase.RUNNING);
        updateWorkerStatus(Map.of(), Map.of("88", running));

        assertEquals(0, endpoint.evictExpiredRequests(10));
        assertEquals(1, endpoint.getInflightRouteRequestCount());
    }

    @Test
    void requestTtlEvictionReleasesCapacityExactlyOnce() throws Exception {
        assertTrue(endpoint.tryCommitRequest(99L, 1_000, 1));
        Thread.sleep(20);

        assertEquals(1, endpoint.evictExpiredInflight(1));
        assertEquals(0, endpoint.evictExpiredInflight(1));
        assertEquals(0, endpoint.getInflightRequestCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());
        assertTrue(endpoint.tryCommitRequest(100L, 1_000, 1));
    }

    @Test
    void engineFenceProtectionPinsExpiredRequestUntilExplicitEnd() {
        assertTrue(endpoint.tryCommitRequest(109L, 1_000, 1));
        assertTrue(endpoint.beginEngineFenceProtection(109L));
        assertTrue(endpoint.beginEngineFenceProtection(109L),
                "begin is idempotent for the same live request");

        assertEquals(0, endpoint.evictExpiredRequests(-1),
                "even an immediately-expired entry remains owned by the fence");
        assertEquals(1, endpoint.getInflightRouteRequestCount());

        assertTrue(endpoint.endEngineFenceProtection(109L));
        assertFalse(endpoint.endEngineFenceProtection(109L));
        assertEquals(1, endpoint.evictExpiredRequests(-1));
        assertEquals(0, endpoint.getInflightRequestCount());
    }

    @Test
    void releaseAndWorkerTerminalClearEngineFenceProtectionExactlyOnce() {
        assertTrue(endpoint.tryCommitRequest(110L, 1_000, 2));
        assertTrue(endpoint.beginEngineFenceProtection(110L));
        assertTrue(endpoint.releaseRequest(110L));
        assertFalse(endpoint.endEngineFenceProtection(110L));

        assertTrue(endpoint.tryCommitRequest(111L, 1_000, 2));
        assertTrue(endpoint.beginEngineFenceProtection(111L));
        updateWorkerStatus(Map.of(
                "111", taskInfo(111L, -1L, TaskPhase.RUNNING)), Map.of());

        assertFalse(endpoint.endEngineFenceProtection(111L));
        assertEquals(0, endpoint.getInflightRequestCount());
        assertEquals(0, endpoint.getInflightRouteRequestCount());
    }

    @Test
    void pendingCountDoesNotDoubleCountTrackedEngineRequest() {
        assertTrue(endpoint.tryCommitRequest(301L, 1_000, 4));

        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(Map.of());
        response.setRunningTaskInfo(Map.of(
                "301", taskInfo(301L, -1L, TaskPhase.RUNNING),
                "302", taskInfo(302L, -1L, TaskPhase.RUNNING)));
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);

        assertEquals(2, endpoint.realPendingCount(),
                "one local request plus one distinct Engine-only request");
    }

    @Test
    void waitEstimateAndMetricsIncludeRequestLedger() throws Exception {
        assertTrue(endpoint.tryCommitRequest(401L, 2_000, 4));
        assertEquals(2_000, endpoint.realWaitTimeMs());
        Thread.sleep(20);

        endpoint.reportBatchMetrics(reporter);

        assertEquals(0, monitor.value(INFLIGHT_BATCH_COUNT));
        assertEquals(1, monitor.value(INFLIGHT_REQUEST_COUNT));
        assertTrue(monitor.value(INFLIGHT_MAX_AGE_MS) >= 10);
    }

    @Test
    void requestWaitAggregateUpdatesWithoutScanningEntryState() throws Exception {
        assertTrue(endpoint.tryCommitRequest(501L, 1_000, 4));
        assertTrue(endpoint.tryCommitRequest(502L, 2_000, 4));
        assertEquals(3_000, endpoint.realWaitTimeMs());

        assertTrue(endpoint.releaseRequest(501L));
        assertEquals(2_000, endpoint.realWaitTimeMs());

        updateWorkerStatus(Map.of(), Map.of(
                "502", taskInfo(502L, -1L, TaskPhase.RUNNING)));
        Thread.sleep(20);
        assertTrue(endpoint.realWaitTimeMs() < 2_000,
                "observed execution progress should reduce the aggregate wait estimate");
    }

    @Test
    void waitSnapshotRetriesAcrossRunningToQueuedTransfer() throws Exception {
        assertTrue(endpoint.tryCommitRequest(503L, 1_000, 4));
        updateWorkerStatus(Map.of(), Map.of(
                "503", taskInfo(503L, -1L, TaskPhase.RUNNING)));

        CountDownLatch queuedContributionRead = new CountDownLatch(1);
        CountDownLatch resumeReader = new CountDownLatch(1);
        AtomicBoolean pauseOnce = new AtomicBoolean();
        endpoint.setWaitSnapshotHookForTest(stage -> {
            if (stage == PrefillEndpoint.WaitSnapshotStage.AFTER_REQUEST_QUEUED_READ
                    && pauseOnce.compareAndSet(false, true)) {
                queuedContributionRead.countDown();
                await(resumeReader);
            }
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Long> waitRead = executor.submit(endpoint::realWaitTimeMs);
            assertTrue(queuedContributionRead.await(5, TimeUnit.SECONDS));

            // The reader has observed queued=0. Move the same request from the
            // running aggregate back to its stripe before it reads running=0.
            updateWorkerStatus(Map.of(), Map.of(
                    "503", taskInfo(503L, -1L, TaskPhase.PENDING)));
            resumeReader.countDown();

            assertTrue(waitRead.get(5, TimeUnit.SECONDS) > 0,
                    "the torn zero snapshot must be rejected and retried");
        } finally {
            resumeReader.countDown();
            endpoint.setWaitSnapshotHookForTest(null);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void staleCachePublicationCannotOverwriteCommitInvalidation() throws Exception {
        CountDownLatch validatedEmptySnapshot = new CountDownLatch(1);
        CountDownLatch resumePublisher = new CountDownLatch(1);
        AtomicBoolean pauseOnce = new AtomicBoolean();
        endpoint.setWaitSnapshotHookForTest(stage -> {
            if (stage == PrefillEndpoint.WaitSnapshotStage.BEFORE_CACHE_PUBLISH
                    && pauseOnce.compareAndSet(false, true)) {
                validatedEmptySnapshot.countDown();
                await(resumePublisher);
            }
        });

        ExecutorService executor = Executors.newSingleThreadExecutor();
        try {
            Future<Long> staleRead = executor.submit(endpoint::realWaitTimeMs);
            assertTrue(validatedEmptySnapshot.await(5, TimeUnit.SECONDS));

            assertTrue(endpoint.tryCommitRequest(504L, 1_000, 4));
            resumePublisher.countDown();
            assertTrue(staleRead.get(5, TimeUnit.SECONDS) > 0,
                    "the combined snapshot must reject a late stale cache publication");
            endpoint.setWaitSnapshotHookForTest(null);
            assertTrue(endpoint.realWaitTimeMs() > 0,
                    "a subsequent read must reject the stale zero cache");
        } finally {
            resumePublisher.countDown();
            endpoint.setWaitSnapshotHookForTest(null);
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void runningWaitDoesNotDependOnRequestIdStripe() throws Exception {
        // 801 and 833 share the low five hash bits; 901 and 902 do not.
        long sameStripeWaitMs = observeRunningPairWait(801L, 833L);
        long differentStripeWaitMs = observeRunningPairWait(901L, 902L);

        assertTrue(Math.abs(sameStripeWaitMs - differentStripeWaitMs) < 100,
                "equivalent running work must not age once per occupied hash stripe: same="
                        + sameStripeWaitMs + ", different=" + differentStripeWaitMs);
    }

    @Test
    void runningServiceCreditStaysWithOldestRequestWhenItFinishesFirst() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(ledger.tryAcquire(1L, 1_000, 0));
        assertTrue(ledger.observe(1L, true, 0));
        clock.set(900);
        assertTrue(ledger.tryAcquire(2L, 1_000, 0));
        assertTrue(ledger.observe(2L, true, 900));
        clock.set(950);
        assertTrue(ledger.settle(1L));

        assertEquals(1_000, ledger.estimate(950),
                "the old request's 950ms service credit must not consume newer work");
    }

    @Test
    void removingNewerRunningRequestPreservesOldestRemainder() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(ledger.tryAcquire(1L, 1_000, 0));
        assertTrue(ledger.observe(1L, true, 0));
        clock.set(900);
        assertTrue(ledger.tryAcquire(2L, 1_000, 0));
        assertTrue(ledger.observe(2L, true, 900));
        clock.set(950);
        assertTrue(ledger.settle(2L));

        assertEquals(50, ledger.estimate(950),
                "removing non-head work must leave the serviced head remainder");
    }

    @Test
    void runningServiceCreditCanCrossMultipleEntries() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(ledger.tryAcquire(1L, 100, 0));
        assertTrue(ledger.tryAcquire(2L, 200, 0));
        assertTrue(ledger.tryAcquire(3L, 300, 0));
        assertTrue(ledger.observe(1L, true, 0));
        assertTrue(ledger.observe(2L, true, 0));
        assertTrue(ledger.observe(3L, true, 0));
        clock.set(250);
        assertTrue(ledger.settle(3L));

        assertEquals(50, ledger.estimate(250),
                "250ms of service must exhaust the first and consume 150ms of the second");
    }

    @Test
    void exhaustedRunningEntryCanBeRemovedIdempotently() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        assertTrue(ledger.tryAcquire(1L, 100, 0));
        assertTrue(ledger.tryAcquire(2L, 200, 0));
        assertTrue(ledger.observe(1L, true, 0));
        assertTrue(ledger.observe(2L, true, 0));
        clock.set(150);
        assertTrue(ledger.tryAcquire(3L, 100, 0));
        assertTrue(ledger.observe(3L, true, 150));
        assertEquals(250, ledger.estimate(150));

        assertTrue(ledger.settle(1L));
        assertFalse(ledger.settle(1L));

        assertEquals(250, ledger.estimate(150),
                "removing an already-exhausted entry must not unlink another request");
    }

    @Test
    void shuffledTenThousandRunningRemovalsUnlinkExactlyOnce() {
        AtomicLong clock = new AtomicLong();
        PrefillRequestLedger ledger = requestLedger(clock);
        List<Long> requests = new ArrayList<>(10_000);
        for (int i = 0; i < 10_000; i++) {
            long requestId = i + 1L;
            requests.add(requestId);
            assertTrue(ledger.tryAcquire(requestId, 1_000_000, 0));
            assertTrue(ledger.observe(requestId, true, 0));
        }
        Collections.shuffle(requests, new Random(0x5eedL));

        for (long requestId : requests) {
            assertTrue(ledger.settle(requestId));
            assertFalse(ledger.settle(requestId));
        }

        assertEquals(0, ledger.estimate(0));
        assertEquals(0, ledger.count());
    }

    @Test
    void runningTransitionAndTtlEvictionCannotDoubleRelease() throws Exception {
        ExecutorService executor = Executors.newFixedThreadPool(2);
        try {
            for (long requestId = 20_000; requestId < 20_050; requestId++) {
                assertTrue(endpoint.tryCommitRequest(requestId, 1_000, 1));
                Thread.sleep(2);

                CountDownLatch start = new CountDownLatch(1);
                CountDownLatch done = new CountDownLatch(2);
                long id = requestId;
                executor.execute(() -> {
                    await(start);
                    updateWorkerStatus(Map.of(), Map.of(
                            Long.toString(id), taskInfo(id, -1L, TaskPhase.RUNNING)));
                    done.countDown();
                });
                executor.execute(() -> {
                    await(start);
                    endpoint.evictExpiredRequests(1);
                    done.countDown();
                });
                start.countDown();
                assertTrue(done.await(5, TimeUnit.SECONDS));

                int live = endpoint.getInflightRouteRequestCount();
                assertTrue(live == 0 || live == 1);
                assertEquals(live, endpoint.getInflightRequestCount());
                if (live == 1) {
                    assertTrue(endpoint.releaseRequest(id));
                }
                assertEquals(0, endpoint.getInflightRouteRequestCount());
                assertEquals(0, endpoint.getInflightRequestCount());
            }
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void legacyEmptyBatchPredictionKeepsExistingAgingSemantics() throws Exception {
        endpoint.commitBatch(700L, 1_000, List.of());
        Thread.sleep(20);

        long waitMs = endpoint.realWaitTimeMs();
        assertTrue(waitMs > 0 && waitMs < 1_000);
    }

    @Test
    void oldRunningWorkDoesNotConsumeNewQueuedRequestPrediction() throws Exception {
        assertTrue(endpoint.tryCommitRequest(601L, 20, 4));
        updateWorkerStatus(Map.of(), Map.of(
                "601", taskInfo(601L, -1L, TaskPhase.RUNNING)));
        Thread.sleep(40);

        assertTrue(endpoint.tryCommitRequest(602L, 5_000, 4));
        assertTrue(endpoint.realWaitTimeMs() >= 4_900,
                "elapsed time before a queued request arrived must not be charged to it");
    }

    @Test
    void agedBatchDoesNotConsumeNewRequestPrediction() throws Exception {
        endpoint.commitBatch(800L, 20, List.of());
        Thread.sleep(40);

        assertTrue(endpoint.tryCommitRequest(701L, 5_000, 4));
        assertTrue(endpoint.realWaitTimeMs() >= 4_900,
                "batch and request progress anchors must remain independent");
    }

    private void updateWorkerStatus(Map<String, TaskInfo> finished,
                                    Map<String, TaskInfo> running) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setFinishedTaskInfo(finished);
        response.setRunningTaskInfo(running);
        endpoint.onWorkerStatusUpdate(endpoint.getStatus(), response);
    }

    private long observeRunningPairWait(long firstRequestId,
                                        long secondRequestId) throws Exception {
        assertTrue(endpoint.tryCommitRequest(firstRequestId, 10_000, 4));
        assertTrue(endpoint.tryCommitRequest(secondRequestId, 10_000, 4));
        updateWorkerStatus(Map.of(), Map.of(
                Long.toString(firstRequestId),
                taskInfo(firstRequestId, -1L, TaskPhase.RUNNING),
                Long.toString(secondRequestId),
                taskInfo(secondRequestId, -1L, TaskPhase.RUNNING)));
        Thread.sleep(200);
        long waitMs = endpoint.realWaitTimeMs();
        assertTrue(endpoint.releaseRequest(firstRequestId));
        assertTrue(endpoint.releaseRequest(secondRequestId));
        return waitMs;
    }

    private BatchItem batchItem(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(512);

        BalanceContext context = new BalanceContext();
        context.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        prefill.setRole(RoleType.PREFILL);
        prefill.setServerIp(endpoint.getIp());
        prefill.setHttpPort(endpoint.getHttpPort());
        prefill.setGrpcPort(endpoint.getGrpcPort());
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(0);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(context, null, null, prefill, null,
                endpoint, null, System.currentTimeMillis());
    }

    private static TaskInfo taskInfo(long requestId, long batchId, TaskPhase phase) {
        TaskInfo task = new TaskInfo();
        task.setRequestId(requestId);
        task.setBatchId(batchId);
        task.setPhase(phase);
        task.setErrorCode(0);
        return task;
    }

    private static void await(CountDownLatch latch) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    private static PrefillRequestLedger requestLedger(AtomicLong clock) {
        return new PrefillRequestLedger(() -> {}, clock::get, ignored -> {});
    }

    private static DecisionGroupHandler noopHandler() {
        return new DecisionGroupHandler() {
            @Override public void onExpired(BatchItem head) {}
            @Override public void onDecisionGroupReady(List<BatchItem> items, DecisionGroupMetadata meta) {}
            @Override public void onOfferFailure(BatchItem item, Throwable error) {}
            @Override public void onDeliveryFailure(BatchItem item, Throwable error) {}
        };
    }

    private static final class RecordingMonitor implements FlexMonitor {
        private final Map<String, Double> values = new HashMap<>();

        @Override
        public void register(String metricName, FlexMetricType metricType) {}

        @Override
        public void register(String metricName,
                             FlexMetricType metricType,
                             FlexPriorityType priorityType) {}

        @Override
        public void register(String metricName, FlexMetricType metricType, int statisticsType) {}

        @Override
        public void report(String metricName, double value) {
            values.put(metricName, value);
        }

        @Override
        public void report(String metricName, FlexMetricTags metricsTags, double value) {
            values.put(metricName, value);
        }

        long value(String metricName) {
            return values.getOrDefault(metricName, -1.0).longValue();
        }
    }
}
