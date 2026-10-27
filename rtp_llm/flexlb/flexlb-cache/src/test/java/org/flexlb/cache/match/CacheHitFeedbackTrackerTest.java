package org.flexlb.cache.match;

import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.AppenderBase;
import com.github.benmanes.caffeine.cache.Cache;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.localstandby.LocalStandbyComparisonService;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.Test;
import org.slf4j.LoggerFactory;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.cache.WorkerStatusTestSupport.workerStatus;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CacheHitFeedbackTrackerTest {
    private final ConfigService configService = new ConfigService("""
            {"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":100000}}}
            """);

    @Test
    void failedTaskFeedbackDoesNotDiscardLaterTasks() {
        WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);
        LocalStandbyComparisonService comparisonService = mock(LocalStandbyComparisonService.class);
        when(comparisonService.captureComparison("broken", RoleType.PREFILL))
                .thenReturn(feedback -> { throw new IllegalStateException("comparison failed"); });
        when(comparisonService.captureComparison("healthy", RoleType.PREFILL))
                .thenReturn(feedback -> CompletableFuture.completedFuture(comparison(worker)));
        CacheHitFeedbackTracker tracker = new CacheHitFeedbackTracker(comparisonService, configService);
        for (String requestId : List.of("broken", "healthy")) {
            tracker.track(requestId, RoleType.PREFILL, "default", worker,
                    100, 0, CacheMatchResult.empty(CacheMatchSource.KVCM), CompletableFuture.completedFuture(null));
        }
        WorkerStatus.TaskTelemetry telemetry = new WorkerStatus.TaskTelemetry(
                true, 1, 2, 3, 4, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0);
        Map<String, WorkerStatus.TaskObservation> tasks = new LinkedHashMap<>();
        for (String requestId : List.of("broken", "healthy")) {
            tasks.put(requestId, new WorkerStatus.TaskObservation(
                    requestId, 0, 0, 100, 0, 0, 0, 0, 0, "", 0,
                    TaskPhase.RUNNING, 0, null, telemetry));
        }
        WorkerStatus.StatusObservation status = mock(WorkerStatus.StatusObservation.class);
        when(status.role()).thenReturn(RoleType.PREFILL);
        when(status.finishedTasks()).thenReturn(tasks);
        when(status.activeTasks()).thenReturn(Map.of());

        var results = tracker.observe(worker, status);

        assertEquals(1, results.size());
        assertTrue(results.getFirst().isDone());
        Cache<?, ?> associations = assertInstanceOf(Cache.class, ReflectionTestUtils.getField(tracker, "predictions"));
        associations.cleanUp();
        assertEquals(0, associations.estimatedSize(), "A failed Finished callback must also release its association");
    }

    @Test
    void concurrentStatusReportsApplyComparisonAndWriteWorkerPvOnce() throws Exception {
        WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);
        LocalStandbyComparisonService comparisonService = mock(LocalStandbyComparisonService.class);
        AtomicInteger comparisons = new AtomicInteger();
        when(comparisonService.captureComparison("request-1", RoleType.PREFILL)).thenReturn(feedback -> {
            comparisons.incrementAndGet();
            return CompletableFuture.completedFuture(comparison(worker));
        });
        CacheHitFeedbackTracker tracker = new CacheHitFeedbackTracker(comparisonService, configService);
        tracker.track("request-1", RoleType.PREFILL, "default", worker,
                100, 0, CacheMatchResult.empty(CacheMatchSource.KVCM), CompletableFuture.completedFuture(null));
        WorkerStatus.TaskTelemetry telemetry = new WorkerStatus.TaskTelemetry(
                true, 1, 2, 3, 4, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0);
        WorkerStatus.TaskObservation task = new WorkerStatus.TaskObservation(
                "request-1", 0, 0, 100, 0, 0, 0, 0, 0, "", 0,
                TaskPhase.RUNNING, 0, null, telemetry);
        WorkerStatus.StatusObservation status = mock(WorkerStatus.StatusObservation.class);
        when(status.role()).thenReturn(RoleType.PREFILL);
        when(status.activeTasks()).thenReturn(Map.of(task.requestId(), task));
        when(status.finishedTasks()).thenReturn(Map.of(task.requestId(), task));
        CountDownLatch firstLogStarted = new CountDownLatch(1);
        CountDownLatch releaseFirstLog = new CountDownLatch(1);
        AtomicInteger workerLogs = new AtomicInteger();
        AppenderBase<ILoggingEvent> appender = new AppenderBase<>() {
            @Override
            protected void append(ILoggingEvent event) {
                if (event.getFormattedMessage().contains("prefill_worker_status")) {
                    if (workerLogs.incrementAndGet() == 1) {
                        firstLogStarted.countDown();
                        try {
                            assertTrue(releaseFirstLog.await(5, TimeUnit.SECONDS));
                        } catch (InterruptedException error) {
                            Thread.currentThread().interrupt();
                            throw new AssertionError(error);
                        }
                    }
                }
            }
        };
        appender.start();
        var logger = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger("pvLogger");
        logger.addAppender(appender);
        try (var executor = Executors.newFixedThreadPool(8)) {
            var first = executor.submit(() -> tracker.observe(worker, status));
            try {
                assertTrue(firstLogStarted.await(2, TimeUnit.SECONDS));
                CountDownLatch otherReportsCompleted = new CountDownLatch(16);
                List<java.util.concurrent.Future<?>> others = new ArrayList<>();
                for (int i = 0; i < 16; i++) {
                    others.add(executor.submit(() -> {
                        try {
                            assertTrue(tracker.observe(worker, status).isEmpty());
                        } finally {
                            otherReportsCompleted.countDown();
                        }
                    }));
                }
                assertTrue(otherReportsCompleted.await(2, TimeUnit.SECONDS),
                        "Duplicate observations must not wait to write another worker PV");
                for (var other : others) {
                    other.get(2, TimeUnit.SECONDS);
                }
            } finally {
                releaseFirstLog.countDown();
            }
            assertEquals(1, first.get(2, TimeUnit.SECONDS).size());
            assertEquals(1, comparisons.get());
            assertEquals(1, workerLogs.get());
        } finally {
            logger.detachAppender(appender);
            appender.stop();
        }
    }
    @Test
    void reportsAvailableSamplesIndependentlyAndOnlyOnce() throws Exception {
        for (boolean predictionAvailable : List.of(false, true)) {
            for (boolean actualAvailable : List.of(false, true)) {
                WorkerStatus worker = workerStatus("127.0.0.1", 8080, RoleType.PREFILL);
                LocalStandbyComparisonService service = mock(LocalStandbyComparisonService.class);
                when(service.captureComparison("request-1", RoleType.PREFILL)).thenReturn(feedback ->
                        CompletableFuture.completedFuture(new org.flexlb.cache.domain.CacheHitComparisonResult(
                                feedback.eventType(), feedback.requestId(), feedback.cacheMatchSource(),
                                feedback.role(), feedback.group(), feedback.workerIdentity(), feedback.taskState(),
                                feedback.inputTokens(), feedback.actualHitTokens(), null,
                                feedback.predictedHitTokens() < 0 ? null
                                        : new org.flexlb.cache.domain.CacheHitComparisonResult.CachePrediction(
                                                feedback.predictedHitTokens(), -1, -1), null, null)));
                CacheHitFeedbackTracker tracker = new CacheHitFeedbackTracker(service, configService);
                CompletableFuture<Long> theory = new CompletableFuture<>();
                tracker.track("request-1", RoleType.PREFILL, "default", worker, 100, 20,
                        predictionAvailable ? CacheMatchResult.empty(CacheMatchSource.KVCM)
                                : CacheMatchResult.failed(CacheMatchSource.KVCM, 10), theory);
                WorkerStatus.TaskTelemetry telemetry = actualAvailable ? new WorkerStatus.TaskTelemetry(
                        true, 1, 2, 3, 4, 5, 0, 6, 0, 0, 0, 0, 0, 0, 0) : null;
                WorkerStatus.TaskObservation task = new WorkerStatus.TaskObservation(
                        "request-1", 0, 0, 100, 40, 0, 0, 0, 0, "", 0,
                        TaskPhase.RUNNING, 0, null, telemetry);
                WorkerStatus.StatusObservation status = mock(WorkerStatus.StatusObservation.class);
                when(status.role()).thenReturn(RoleType.PREFILL);
                when(status.finishedTasks()).thenReturn(Map.of(task.requestId(), task));
                when(status.activeTasks()).thenReturn(Map.of());

                var results = tracker.observe(worker, status);
                assertEquals(1, results.size());
                org.junit.jupiter.api.Assertions.assertFalse(results.getFirst().isDone());
                theory.complete(60L);
                var result = results.getFirst().get(2, TimeUnit.SECONDS);
                assertEquals(60L, result.theoryHitTokens());
                assertEquals(actualAvailable ? task.prefixLength() : -1, result.actualHitTokens());
                assertEquals(predictionAvailable, result.kvcmPrediction() != null);
                assertTrue(tracker.observe(worker, status).isEmpty());
            }
        }
    }

    private static org.flexlb.cache.domain.CacheHitComparisonResult comparison(WorkerStatus worker) {
        return new org.flexlb.cache.domain.CacheHitComparisonResult(
                "cache_hit_comparison", "request-1", "KVCM", "PREFILL", "default",
                worker.getWorkerIdentity(), "FINISHED", 100, 0, null, null, null, null);
    }

}
