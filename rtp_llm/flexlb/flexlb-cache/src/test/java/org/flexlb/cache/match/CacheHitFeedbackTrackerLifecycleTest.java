package org.flexlb.cache.match;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import ch.qos.logback.core.read.ListAppender;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Ticker;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.cache.CacheMatchTestConfigurations.localSync;
import static org.flexlb.cache.WorkerStatusTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mockStatic;

class CacheHitFeedbackTrackerLifecycleTest {
    private final Logger pvLogger = (Logger) LoggerFactory.getLogger("pvLogger");
    private final ListAppender<ILoggingEvent> pv = new ListAppender<>();
    private final WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);
    private CacheHitFeedbackTracker tracker = new CacheHitFeedbackTracker(
            new LocalStandbyComparisonService(localSync(new ModelMetaConfig()), null), new ConfigService("""
                    {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":100000}}}
                    """));

    @BeforeEach
    void captureStatusPv() {
        pv.start();
        pvLogger.addAppender(pv);
    }

    @AfterEach
    void stopCapturingStatusPv() {
        pvLogger.detachAppender(pv);
        pv.stop();
    }

    @Test
    void finishedFeedbackReleasesItsAssociation() {
        track("finished", 1000, CompletableFuture.completedFuture(null));

        var results = observe(task("finished", 1000, 700, true, 6), true);

        assertEquals(1, results.size());
        assertEquals(700, results.getFirst().join().actualHitTokens());
        assertEquals(0, associationCount());
    }

    @Test
    void runningWithoutFinishedExpiresAfterFeedbackSilence() {
        AtomicLong timeNs = new AtomicLong();
        tracker = trackerWithClock(timeNs);
        track("missing-finished", 1000, CompletableFuture.completedFuture(null));
        assertEquals(700,
                observe(task("missing-finished", 1000, 700, true, 0), false).getFirst().join().actualHitTokens());

        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(900));
        assertEquals(1, associationCount(), "WorkerStatus timeout must not control feedback retention");
        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(100));

        assertEquals(0, associationCount());
        assertTrue(observe(task("missing-finished", 1000, 700, true, 6), true).isEmpty());
        assertTrue(pv.list.isEmpty(), "Expired feedback must not generate a late status PV");
    }

    @Test
    void unreportedPredictionExpiresFromRegistration() {
        AtomicLong timeNs = new AtomicLong();
        tracker = trackerWithClock(timeNs);
        track("unreported", 1000, CompletableFuture.completedFuture(null));

        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(999));
        assertEquals(1, associationCount());
        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(1));

        assertEquals(0, associationCount());
        assertTrue(observe(task("unreported", 1000, 700, true, 6), true).isEmpty());
        assertTrue(pv.list.isEmpty());
    }

    @Test
    void matchingRunningFeedbackRenewsRetentionWithoutRepeatingComparison() {
        AtomicLong timeNs = new AtomicLong();
        tracker = trackerWithClock(timeNs);
        track("long-running", 1000, CompletableFuture.completedFuture(null));
        var running = task("long-running", 1000, 700, true, 0);
        assertEquals(700, observe(running, false).getFirst().join().actualHitTokens());

        for (int round = 0; round < 3; round++) {
            timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(900));
            assertTrue(observe(running, false).isEmpty());
            assertEquals(1, associationCount());
        }
        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(900));

        assertTrue(observe(task("long-running", 1000, 700, true, 6), true).isEmpty());
        assertEquals(1, pv.list.size());
        assertEquals(0, associationCount());
    }

    @ParameterizedTest
    @EnumSource(value = TaskPhase.class, names = {"PENDING", "RECEIVED", "KV_ALLOCATED"})
    void matchingWaitingFeedbackRenewsRetentionUntilFinished(TaskPhase phase) {
        AtomicLong timeNs = new AtomicLong();
        tracker = trackerWithClock(timeNs);
        track("waiting", 1000, CompletableFuture.completedFuture(null));
        var waiting = new WorkerStatus.TaskObservation("waiting", 700, 0, 1000, 0, 0, 0, 0, 0, "", 0, phase, 0, null,
                task("waiting", 1000, 700, false, 0).telemetry());
        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(900));

        assertTrue(observe(waiting, false).isEmpty());
        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(900));
        assertEquals(1, associationCount());
        assertEquals(700, observe(task("waiting", 1000, 700, true, 6), true).getFirst().join().actualHitTokens());
        assertEquals(1, pv.list.size());
        assertEquals(0, associationCount());
    }

    @Test
    void anotherRequestsFeedbackDoesNotRenewMissingRequest() {
        AtomicLong timeNs = new AtomicLong();
        tracker = trackerWithClock(timeNs);
        track("missing", 1000, CompletableFuture.completedFuture(null));
        track("active", 1000, CompletableFuture.completedFuture(null));
        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(900));
        assertEquals(700, observe(task("active", 1000, 700, true, 0), false).getFirst().join().actualHitTokens());
        timeNs.addAndGet(TimeUnit.MILLISECONDS.toNanos(100));

        assertEquals(1, associationCount());
        assertTrue(observe(task("missing", 1000, 700, true, 6), true).isEmpty());
        assertTrue(pv.list.isEmpty());
        assertTrue(observe(task("active", 1000, 700, true, 6), true).isEmpty());
        assertEquals(1, pv.list.size());
        assertEquals(0, associationCount());
    }

    private static CacheHitFeedbackTracker trackerWithClock(AtomicLong timeNs) {
        try (var tickers = mockStatic(Ticker.class)) {
            tickers.when(Ticker::systemTicker).thenReturn((Ticker) timeNs::get);
            return new CacheHitFeedbackTracker(
                    new LocalStandbyComparisonService(localSync(new ModelMetaConfig()), null), new ConfigService(
                            """
                                    {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":100000,"statusUpdateTimeoutMs":7}},
                                    "observability":{"cacheHit":{"feedbackTimeoutMs":1000}}}
                                    """));
        }
    }

    @Test
    void runningKeepsAssociationForFinishedStatusPvWithoutRepeatingComparison() {
        track("running", 1000, CompletableFuture.completedFuture(null));
        var running = task("running", 1000, 700, true, 0);

        assertEquals(700, observe(running, false).getFirst().join().actualHitTokens());
        assertEquals(1, associationCount());
        assertTrue(pv.list.isEmpty());
        assertTrue(observe(running, false).isEmpty());

        var finished = task("running", 1000, 700, true, 6);
        assertTrue(observe(finished, true).isEmpty());
        assertEquals(1, pv.list.size());
        assertEquals(0, associationCount());
        assertTrue(observe(finished, true).isEmpty());
        assertTrue(observe(running, false).isEmpty());
        assertEquals(1, pv.list.size());
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void finishedWithoutValidActualHitStillReleasesAssociation(boolean hasTelemetry) {
        track("unavailable", 1000, CompletableFuture.completedFuture(600L));
        var telemetry = hasTelemetry ? task("unavailable", 1000, 700, false, 0).telemetry() : null;
        var finished = new WorkerStatus.TaskObservation("unavailable", 700, 0, 1000, 0, 0, 0, 0, 0, "", 0,
                TaskPhase.RUNNING, 0, null, telemetry);

        var result = observe(finished, true).getFirst().join();

        assertEquals(-1, result.actualHitTokens());
        assertEquals(600, result.theoryHitTokens());
        assertEquals(100, result.kvcmPrediction().predictedHitTokens());
        assertEquals(0, associationCount());
    }

    @Test
    void finishedCleanupDoesNotCancelPendingComparison() {
        var theory = new CompletableFuture<Long>();
        track("async", 1000, theory);
        var finished = task("async", 1000, 700, true, 6);

        var comparison = observe(finished, true).getFirst();

        assertFalse(comparison.isDone());
        assertEquals(0, associationCount());
        assertTrue(observe(finished, true).isEmpty());
        theory.complete(600L);
        assertEquals(700, comparison.join().actualHitTokens());
        assertEquals(600, comparison.join().theoryHitTokens());
        assertEquals(1, pv.list.size());
    }

    @ParameterizedTest
    @CsvSource({"false,true", "true,false", "true,true"})
    void concurrentFeedbackSharesComparisonAndStatusPvClaims(boolean firstFinished, boolean secondFinished)
            throws Exception {
        track("concurrent", 1000, CompletableFuture.completedFuture(null));
        var engineTask = task("concurrent", 1000, 700, true, 6);
        var blocker = new BlockingStatusAppender();
        blocker.start();
        pvLogger.addAppender(blocker);
        try (var executor = Executors.newSingleThreadExecutor()) {
            var first = executor.submit(() -> observe(engineTask, firstFinished));
            try {
                assertTrue(blocker.started.await(5, TimeUnit.SECONDS));
                assertTrue(observe(engineTask, secondFinished).isEmpty());
                assertEquals(secondFinished ? 0 : 1, associationCount());
            } finally {
                blocker.release.countDown();
            }
            assertEquals(700, first.get(5, TimeUnit.SECONDS).getFirst().join().actualHitTokens());
            assertEquals(0, associationCount());
            assertTrue(observe(engineTask, true).isEmpty());
            assertTrue(observe(engineTask, false).isEmpty());
            assertEquals(1, pv.list.size());
        } finally {
            pvLogger.detachAppender(blocker);
            blocker.stop();
        }
    }

    @Test
    void oldFinishedCallbackDoesNotRemoveReplacementAssociation() throws Exception {
        track("replaced", 1000, CompletableFuture.completedFuture(null));
        var blocker = new BlockingStatusAppender();
        blocker.start();
        pvLogger.addAppender(blocker);
        try (var executor = Executors.newSingleThreadExecutor()) {
            var oldFinished = executor.submit(() -> observe(task("replaced", 1000, 700, true, 6), true));
            try {
                assertTrue(blocker.started.await(5, TimeUnit.SECONDS));
                track("replaced", 2000, CompletableFuture.completedFuture(null));
            } finally {
                blocker.release.countDown();
            }
            assertEquals(700, oldFinished.get(5, TimeUnit.SECONDS).getFirst().join().actualHitTokens());
            assertEquals(1, associationCount());
            assertEquals(900, observe(task("replaced", 2000, 900, true, 6), true).getFirst().join().actualHitTokens());
            assertEquals(0, associationCount());
            assertEquals(2, pv.list.size());
        } finally {
            pvLogger.detachAppender(blocker);
            blocker.stop();
        }
    }

    @Test
    void capacityPressurePreservesEveryEngineCacheHitRecordAndReleasesFinishedHistory() {
        Map<String, EngineCacheHit> engineCacheHits = new LinkedHashMap<>();
        List<EngineCacheHit> fixtures = List.of(new EngineCacheHit(1000, 0), new EngineCacheHit(449347, 448000),
                new EngineCacheHit(78781, 77056), new EngineCacheHit(2048, 0));
        for (int index = 0; index < 120_000; index++) {
            engineCacheHits.put("engine-" + index, fixtures.get(index % fixtures.size()));
        }
        Map<String, EngineCacheHit> masterCacheHits = new HashMap<>();
        var pending = new ArrayDeque<WorkerStatus.TaskObservation>();
        Level previousLevel = pvLogger.getLevel();
        pvLogger.setLevel(Level.OFF);
        try {
            int index = 0;
            for (var entry : engineCacheHits.entrySet()) {
                var engineHit = entry.getValue();
                track(entry.getKey(), engineHit.inputTokens(), CompletableFuture.completedFuture(null));
                pending.add(task(entry.getKey(), engineHit.inputTokens(), engineHit.hitTokens(), true, 6));
                if (pending.size() > 256) {
                    collectEngineFeedback(pending.remove(), masterCacheHits, index % 2 == 0);
                }
                if (++index % 4096 == 0) {
                    assertEquals(pending.size(), associationCount(),
                            "Finished history must not consume pending capacity");
                }
            }
            while (!pending.isEmpty()) {
                collectEngineFeedback(pending.remove(), masterCacheHits, true);
            }
        } finally {
            pvLogger.setLevel(previousLevel);
        }

        assertEquals(engineCacheHits, masterCacheHits);
        assertEquals(engineCacheHits.values().stream().mapToLong(EngineCacheHit::hitTokens).sum(),
                masterCacheHits.values().stream().mapToLong(EngineCacheHit::hitTokens).sum());
        assertEquals(engineCacheHits.values().stream().mapToLong(EngineCacheHit::inputTokens).sum(),
                masterCacheHits.values().stream().mapToLong(EngineCacheHit::inputTokens).sum());
        assertEquals(0, associationCount());
    }

    private void collectEngineFeedback(WorkerStatus.TaskObservation engineTask,
                                       Map<String, EngineCacheHit> masterCacheHits,
                                       boolean runningFirst) {
        if (runningFirst) {
            collect(observe(engineTask, false), masterCacheHits);
        }
        collect(observe(engineTask, true), masterCacheHits);
        assertTrue(observe(engineTask, true).isEmpty());
        assertTrue(observe(engineTask, false).isEmpty());
    }

    private static void collect(List<CompletableFuture<CacheHitComparisonResult>> results,
                                Map<String, EngineCacheHit> masterCacheHits) {
        for (var future : results) {
            var comparison = future.join();
            assertNull(
                    masterCacheHits.put(comparison.requestId(),
                            new EngineCacheHit(comparison.inputTokens(), comparison.actualHitTokens())),
                    "The Master must not count the same engine feedback twice");
        }
    }

    private void track(String requestId, long inputTokens, CompletableFuture<Long> theory) {
        tracker.track(requestId, RoleType.PREFILL, "default", worker, inputTokens, 100,
                CacheMatchResult.empty(CacheMatchSource.KVCM), theory);
    }

    private List<CompletableFuture<CacheHitComparisonResult>> observe(WorkerStatus.TaskObservation task,
                                                                      boolean finished) {
        var status = worker.bindStatusObservation(worker.committedEngineObservation(), true, 1L, 1L,
                finished ? Map.of(task.requestId(), task) : Map.of());
        if (!finished) {
            var engine = worker.committedEngineObservation();
            status = worker.bindStatusObservation(
                    new WorkerStatus.EngineObservation(engine.role(), engine.availableConcurrency(),
                            engine.availableKvCacheTokens(), engine.totalKvCacheTokens(),
                            Map.of(task.requestId(), task), engine.stepLatencyMs(), engine.iterateCount(),
                            engine.dpSize(), engine.tpSize(), engine.dpRank(), engine.dpMasterAddress(),
                            engine.blockSize(), engine.blockHashLookaheadTokens(), engine.cacheMatchRollbackBlocks(),
                            engine.kvCacheGroupMode(), engine.maxSeqLen(), engine.maxBatchTokensSize(),
                            engine.runningQueryLen(), engine.waitingQueryLen(), engine.lastStepMetrics()),
                    true, 1L, 1L, Map.of());
        }
        return tracker.observe(worker, status);
    }

    private static WorkerStatus.TaskObservation task(String requestId,
                                                     long inputTokens,
                                                     long hitTokens,
                                                     boolean valid,
                                                     long firstTokenTimeMs) {
        var telemetry =
                new WorkerStatus.TaskTelemetry(valid, 1, 2, 3, 4, 5, 0, firstTokenTimeMs, hitTokens, 0, 0, 0, 0, 0, 0);
        return new WorkerStatus.TaskObservation(requestId, hitTokens, 0, inputTokens, 0, 0, 0, 0, 0, "", 0,
                TaskPhase.RUNNING, 0, null, telemetry);
    }

    private long associationCount() {
        Cache<?, ?> associations = assertInstanceOf(Cache.class, ReflectionTestUtils.getField(tracker, "predictions"));
        associations.cleanUp();
        return associations.estimatedSize();
    }

    private record EngineCacheHit(long inputTokens, long hitTokens) {
    }

    private static final class BlockingStatusAppender extends AppenderBase<ILoggingEvent> {
        private final CountDownLatch started = new CountDownLatch(1);
        private final CountDownLatch release = new CountDownLatch(1);
        private final AtomicBoolean first = new AtomicBoolean(true);

        @Override
        protected void append(ILoggingEvent event) {
            if (first.compareAndSet(true, false)) {
                started.countDown();
                try {
                    assertTrue(release.await(5, TimeUnit.SECONDS));
                } catch (InterruptedException interrupted) {
                    Thread.currentThread().interrupt();
                    throw new AssertionError(interrupted);
                }
            }
        }
    }
}
