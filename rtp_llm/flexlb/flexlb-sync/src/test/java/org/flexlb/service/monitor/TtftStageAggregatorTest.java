package org.flexlb.service.monitor;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

class TtftStageAggregatorTest {
    private static final long BASE = 1_700_000_000_000L;
    private static final String ID = "frontend";
    private static final String ATTEMPT = ID + "#call0123456789abcdef0123456789abcdef";
    private static final FlexMetricTags TAGS = FlexMetricTags.of("deployment_name", "model-a");
    private final FlexMonitor monitor = mock(FlexMonitor.class);
    private final AtomicLong clock = new AtomicLong(BASE + 1000);
    private TtftStageAggregator aggregator = create(100, 120000, false);

    @Test
    void registersStageAndFailureCounters() {
        for (String metric : TtftStageReporter.METRICS) {
            verify(monitor).register(TtftStageReporter.PREFIX + metric, FlexMetricType.COUNTER,
                    FlexPriorityType.NORMAL);
        }
    }

    @Test
    void joinsAllThreeStreamsInAnyArrivalOrderAndWaitsForGrace() {
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, 5000L, 10000L, 30000L, 2000L, 4));
        receipt(ATTEMPT, 10000);
        dispatch(ATTEMPT, 0);
        aggregator.publishReady();
        none("requests");
        publish();
        value("total.us", 80000);
        value("before.prefill.us", 10000);
        value("kv.alloc.us", 5000);
        value("remote.kv.us", 0);
        value("scheduler.wait.us", 10000);
        value("prefill.inflight.us", 30000);
        value("other.us", 25000);
        value("requests", 1);
        value("ple.wait.us", 2000);
        value("ple.requests", 1);
    }

    @Test
    void acceptsIdsAndDeploymentNamesUpToTheDaemonLimit() {
        String id = "r".repeat(2048);
        String deployment = "d".repeat(2048);
        aggregator.addTiming("fe", id, "frontend", deployment, BASE * 1000, null);
        receipt(id, 10000);
        aggregator.addWorkerStatus("worker", task(id, 10, 80, false, 0L, 1000L, 1000L, null, 0));
        publish();
        verify(monitor).report(TtftStageReporter.PREFIX + "requests", FlexMetricTags.of("deployment_name", deployment),
                1);
    }

    @Test
    void attributesRemoteKvOnlyWithExplicitPreSchedulerMode() {
        aggregator = create(100, 120000, true);
        sample();
        publish();
        value("remote.kv.us", 4000);
        value("scheduler.wait.us", 10000);
        value("other.us", 21000);
    }

    @Test
    void missingIndividualStagesLeaveMeasuredStagesAndResidual() {
        dispatch(ATTEMPT, 0);
        receipt(ATTEMPT, 10000);
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, null, 10000L, null, null, 0));
        publish();
        value("kv.alloc.us", 0);
        value("scheduler.wait.us", 10000);
        value("prefill.inflight.us", 0);
        value("other.us", 60000);
        value("requests", 1);
        none("ple.requests");
    }

    @Test
    void missingAllStageTelemetryDoesNotFabricateAnAllOtherSample() {
        dispatch(ATTEMPT, 0);
        receipt(ATTEMPT, 10000);
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, null, null, null, 2000L, 0));
        publish();
        none("requests");
        value("ple.wait.us", 2000);
        value("ple.requests", 1);
    }

    @Test
    void zeroStagesAreMeasuredAndCounted() {
        dispatch(ATTEMPT, 0);
        receipt(ATTEMPT, 10000);
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, 0L, 0L, 0L, 0L, 0));
        publish();
        value("other.us", 70000);
        value("requests", 1);
        value("ple.wait.us", 0);
    }

    @Test
    void rejectsNegativeOverflowAndNonClosingMeasuredDurations() {
        for (long allocation : new long[]{-1, Long.MAX_VALUE, 100001}) {
            aggregator = create(100, 120000, false);
            dispatch(ATTEMPT, 0);
            receipt(ATTEMPT, 10000);
            aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, allocation, 10000L, 30000L, null, 0));
            publish();
        }
        none("requests");
    }

    @Test
    void waitsForMissingWorkerAndRestartsGraceWhenItArrives() {
        dispatch(ATTEMPT, 0);
        receipt(ATTEMPT, 10000);
        publish();
        none("requests");
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, 0L, 10000L, 30000L, null, 0));
        aggregator.publishReady();
        none("requests");
        publish();
        value("requests", 1);
    }

    @Test
    void conflictingRepeatedObservationCannotChooseAnArbitraryWinner() {
        sample();
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 90, false, 5000L, 10000L, 30000L, 2000L, 4));
        publish();
        none("requests");
    }

    @Test
    void sourceGapAndLeaderResetCannotRebuildPartialOldHistory() {
        sample();
        aggregator.clearSource("worker");
        sample();
        publish();
        none("requests");
        aggregator.resetAll();
        sample();
        publish();
        none("requests");
    }

    @Test
    void sourceGapDiscardsOnlyAssociatedRequestsAndKeepsHealthySources() {
        sample();
        healthySample("healthy", 200);
        aggregator.clearSource("worker");
        sample();
        publish();
        value("requests", 1);
        value("total.us", 150000);
        verify(monitor, never()).report(TtftStageReporter.PREFIX + "total.us", TAGS, 80000);
    }

    @Test
    void clearingUnknownSourceDoesNotChangeTheObservationWindow() {
        aggregator.clearSource("new-source");
        sample();
        publish();
        value("requests", 1);
    }

    @Test
    void sourceGapPreservesPublishedDedupAndAcceptsOtherRetainedRequests() {
        sample();
        publish();
        aggregator.clearSource("worker");
        sample();
        healthySample("healthy", 200);
        publish();
        verify(monitor, times(2)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
        value("total.us", 80000);
        value("total.us", 150000);
    }

    @Test
    void rejectedIdenticalReplayKeepsItsRetainedWorkerObservation() {
        sample();
        aggregator.discardWorkerStatus("worker",
                List.of(task(ATTEMPT, 10, 80, false, 5000L, 10000L, 30000L, 2000L, 4)));
        publish();
        value("requests", 1);
        value("total.us", 80000);
    }

    @Test
    void producerInvalidationCancelsAnAlreadyCopiedPublishSnapshot() throws Exception {
        sample();
        healthySample("healthy", 200);
        try (var producer = Executors.newSingleThreadExecutor()) {
            doAnswer(invocation -> {
                double total = invocation.getArgument(2);
                String lostId = total == 80000 ? "healthy" : ATTEMPT;
                long start = total == 80000 ? 220 : 10;
                long end = total == 80000 ? 350 : 80;
                producer.submit(() -> aggregator.discardWorkerStatus("dropped-worker",
                        List.of(task(lostId, start, end, true, 0L, 1000L, 2000L, null, 0)))).get(5, TimeUnit.SECONDS);
                return null;
            }).when(monitor).report(eq(TtftStageReporter.PREFIX + "total.us"), eq(TAGS), anyDouble());
            publish();
        }
        verify(monitor, times(1)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
    }

    @Test
    void expirationDoesNotAllowLateReplayToCountAgain() {
        aggregator = create(100, 10000, false);
        sample();
        publish();
        clock.addAndGet(10000);
        aggregator.publishReady();
        sample();
        publish();
        verify(monitor, times(1)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
    }

    @Test
    void overflowingUnrelatedReceiptDoesNotEvictACompleteHealthyRequest() {
        aggregator = create(3, 120000, false);
        sample();
        receipt("unrelated", 50000);
        publish();
        value("requests", 1);
        sample();
        publish();
        value("requests", 1);
    }

    @Test
    void fullPublishedDedupPoolLeavesAReadyRequestInPlaceUntilExpiry() {
        aggregator = create(3, 20000, false);
        for (int index = 0; index < 3; index++) {
            long start = (clock.get() - BASE) * 1000;
            dispatch("completed-" + index, start);
            receipt("completed-" + index, start + 10000);
            aggregator.addWorkerStatus("worker",
                    task("completed-" + index, start / 1000 + 10, start / 1000 + 80, false, 0L, 1000L, 2000L, null, 0));
            publish();
        }
        healthySample("waiting", 10000);
        publish();
        verify(monitor, times(3)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
        clock.set(BASE + 26000);
        aggregator.publishReady();
        verify(monitor, times(4)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
        value("total.us", 150000);
    }

    @Test
    void futureTimestampCannotExtendOrBreakCapacityState() {
        aggregator = create(3, 120000, false);
        aggregator.addTiming("prefill", "future", "prefill", "model-a", Long.MAX_VALUE, null);
        sample();
        publish();
        value("requests", 1);
    }

    @Test
    void multipleRequestsReportRequestWeightedTotalsWithOneCountEach() {
        sample();
        dispatch("second", 200000);
        receipt("second", 220000);
        aggregator.addWorkerStatus("worker", task("second", 220, 350, false, 10000L, 20000L, 60000L, null, 0));
        publish();
        value("total.us", 80000);
        value("total.us", 150000);
        verify(monitor, times(2)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
    }

    @Test
    void windowUsesPairingTimeWhenFrontendTimingArrivesBeforeBoundary() {
        long window = TtftStageQuantiles.WINDOW_MS;
        long windowStart = BASE / window * window;
        clock.set(windowStart);
        aggregator = create(100, 120000, false);
        clock.set(windowStart + window - 1000);
        sample();
        publish();
        value("requests", 1);
        none("window20s.total.us");
        clock.set(windowStart + 2 * window);
        aggregator.publishReady();
        String slot = Long.toString((windowStart / window + 1) % 2);
        FlexMetricTags quantileTags = FlexMetricTags.of("deployment_name", "model-a", "slot", slot, "statistic", "p95");
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "total.us", quantileTags, 80000);
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "other.us", quantileTags, 25000);
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "samples",
                FlexMetricTags.of("deployment_name", "model-a", "slot", slot), 1);
        verify(monitor).report(TtftStageReporter.WINDOW_PREFIX + "window.end.ms",
                FlexMetricTags.of("deployment_name", "model-a", "slot", slot), windowStart + 2 * window);
        verify(monitor, times(1)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
    }

    @Test
    void concurrentPublishersCannotCountTheSameRequestTwice() throws Exception {
        sample();
        clock.addAndGet(3000);
        try (var executor = Executors.newFixedThreadPool(4)) {
            List<java.util.concurrent.Future<?>> futures = new ArrayList<>();
            for (int index = 0; index < 8; index++) {
                futures.add(executor.submit(aggregator::publishReady));
            }
            for (var future : futures) {
                future.get(5, TimeUnit.SECONDS);
            }
        }
        value("requests", 1);
    }

    @Test
    void refusesAWindowThatCannotWaitAcrossOnePoll() {
        assertThrows(IllegalArgumentException.class, () -> create(2, 10000, false));
        assertThrows(IllegalArgumentException.class,
                () -> new TtftStageAggregator(monitor, 100, 1000, 999, false, clock::get));
    }

    @Test
    void retriesWithOverlappingExecutionTimesKeepTheirOwnSixStages() {
        String retry = ATTEMPT + "#attempt1";
        sample();
        receipt(retry, 55_000);
        aggregator.addWorkerStatus("worker", task(retry + "-p-a12b34cd", 56, 100, false, 1000L, 2000L, 37000L, 0L, 0));
        receipt(retry + "-p-a12b34cd", 56_000);
        dispatch(retry, 50_000);
        publish();
        value("total.us", 80_000);
        value("total.us", 50_000);
        value("before.prefill.us", 5_000);
        value("other.us", 5_000);
        verify(monitor, times(2)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
    }

    @Test
    void failedDispatchWithoutWorkerHasOneFailureDurationAndNoTtft() {
        failed(ATTEMPT, 10_000, 35_000);
        failed(ATTEMPT, 10_000, 35_000);
        publish();
        none("failed.requests");
        clock.addAndGet(120000);
        aggregator.publishReady();
        value("failed.us", 25_000);
        value("failed.requests", 1);
        none("requests");
    }

    @Test
    void clientFailsBeforeWorkerToken() {
        failed(ATTEMPT, 0, 25_000);
        receipt(ATTEMPT, 10_000);
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, 0L, 0L, 0L, 0L, 0));
        publish();
        value("failed.us", 25_000);
        none("requests");
    }

    @Test
    void failureAfterTokenPreservesTtftEvenWhenWorkerArrivesLaterWithAnError() {
        failed(ATTEMPT, 0, 90_000);
        receipt(ATTEMPT, 10_000);
        publish();
        none("failed.requests");
        var success = task(ATTEMPT, 10, 80, false, 5000L, 10000L, 30000L, 0L, 0);
        aggregator.addWorkerStatus("worker", new WorkerStatus.TaskObservation(ATTEMPT, 0, 0, 0, 0, 0, BASE + 100, 0, 1,
                "after token", 0, TaskPhase.RUNNING, 0, PriorityPreemptionProgress.NONE, success.telemetry()));
        publish();
        value("total.us", 80_000);
        none("failed.requests");
    }

    @Test
    void workerWithoutFirstTokenDoesNotContributeItsEndTimeToTtft() {
        receipt(ATTEMPT, 10_000);
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, true, 0L, 10000L, 30000L, 0L, 0));
        publish();
        none("requests");
        failed(ATTEMPT, 0, 85_000);
        publish();
        value("failed.us", 85_000);
        none("requests");
    }

    @Test
    void attributesOnlyTokenProducingWorkerUnderThePrefillProxy() {
        String inner = ATTEMPT + "-p-a12b34cd#callfedcba9876543210fedcba9876543210";
        dispatch(ATTEMPT, 0);
        receipt(ATTEMPT, 10_000);
        receipt(inner, 11_000);
        aggregator.addWorkerStatus("p", task(inner, 12, 25, true, 7000L, 1000L, 2000L, 0L, 0));
        aggregator.addWorkerStatus("p", task(inner + "#attempt1", 32, 80, false, 0L, 2000L, 20000L, 0L, 0));
        publish();
        value("total.us", 80_000);
        value("before.prefill.us", 10_000);
        value("kv.alloc.us", 0);
        value("prefill.inflight.us", 20_000);
        value("other.us", 48_000);
        value("requests", 1);
    }

    @Test
    void missingWorkerOnOverflowCannotBeMisclassifiedAsFailure() {
        aggregator = create(3, 20_000, false);
        for (int i = 0; i < 3; i++) {
            healthySample("healthy-" + i, 0);
        }
        aggregator.addWorkerStatus("other-worker", task(ATTEMPT, 10, 20, false, 0L, 0L, 0L, 0L, 0));
        publish();
        failed(ATTEMPT, 0, 90_000);
        clock.addAndGet(24_000);
        aggregator.publishReady();
        none("failed.requests");
        verify(monitor, times(3)).report(TtftStageReporter.PREFIX + "requests", TAGS, 1);
    }

    @Test
    void proxyTokenAfterClientFailureNeverBecomesTtftWhileWaitingForLateWorkerStatus() {
        failed(ATTEMPT, 0, 25_000);
        receipt(ATTEMPT, 10_000);
        aggregator.addWorkerStatus("worker", task(ATTEMPT + "-p-0123abcd", 10, 80, false, 0L, 0L, 0L, 0L, 0));
        publish();
        none("requests");
        none("failed.requests");
        clock.addAndGet(120_000);
        aggregator.publishReady();
        value("failed.us", 25_000);
        none("requests");
    }

    private void failed(String id, long startUs, long endUs) {
        aggregator.addTiming("frontend", id, "frontend", "model-a", BASE * 1000 + startUs, BASE * 1000 + endUs);
    }

    private void sample() {
        dispatch(ATTEMPT, 0);
        receipt(ATTEMPT, 10000);
        aggregator.addWorkerStatus("worker", task(ATTEMPT, 10, 80, false, 5000L, 10000L, 30000L, 2000L, 4));
    }

    private void healthySample(String requestId, long receivedOffsetMs) {
        dispatch(requestId, receivedOffsetMs * 1000);
        receipt(requestId, (receivedOffsetMs + 20) * 1000);
        aggregator.addWorkerStatus("healthy-worker",
                task(requestId, receivedOffsetMs + 20, receivedOffsetMs + 150, false, 10000L, 20000L, 60000L, null, 0));
    }

    private TtftStageAggregator create(int capacity, long retentionMs, boolean preSchedulerKv) {
        return new TtftStageAggregator(monitor, capacity, retentionMs, 3000, preSchedulerKv, clock::get);
    }

    private void dispatch(String id, long receivedOffsetUs) {
        aggregator.addTiming("frontend", id, "frontend", "model-a", BASE * 1000 + receivedOffsetUs, null);
    }

    private void receipt(String id, long receivedOffsetUs) {
        aggregator.addTiming("prefill", id, "prefill", "model-a", BASE * 1000 + receivedOffsetUs, null);
    }

    private void publish() {
        clock.addAndGet(3000);
        aggregator.publishReady();
    }

    private void value(String metric, double value) {
        verify(monitor).report(TtftStageReporter.PREFIX + metric, TAGS, value);
    }

    private void none(String metric) {
        verify(monitor, never()).report(eq(TtftStageReporter.PREFIX + metric), any(FlexMetricTags.class), anyDouble());
    }

    private static WorkerStatus.TaskObservation task(String id,
                                                     long startMs,
                                                     long endMs,
                                                     boolean failed,
                                                     Long allocationUs,
                                                     Long queueUs,
                                                     Long inflightUs,
                                                     Long pleUs,
                                                     long remoteMs) {
        var telemetry =
                new WorkerStatus.TaskTelemetry(false, BASE + startMs, 0, 0, 0, 0, remoteMs, failed ? 0 : BASE + endMs,
                        0, 0, 0, 0, 0, 0, 0, null, null, null, queueUs, null, allocationUs, inflightUs, null, pleUs);
        return new WorkerStatus.TaskObservation(id, 0, 0, 0, 0, 0, BASE + endMs, 0, failed ? 1 : 0, "", 0,
                TaskPhase.RUNNING, 0, PriorityPreemptionProgress.NONE, telemetry);
    }
}
