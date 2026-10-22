package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** TaskInfo fields match the current Master; a prepared zero is not an unknown hit. */
class PrefillHitFeedbackTest {
    @TempDir Path directory;
    private ScheduledExecutorService scheduler;
    private JavaMockEngineCluster.FastRpcService prefill;

    private void start(String durationMs) throws Exception {
        scheduler = Executors.newScheduledThreadPool(3);
        prefill = new JavaMockEngineCluster.FastRpcService("prefill-0",
                EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL, 0, new ConcurrentHashMap<>(),
                scheduler, performanceModel(directory, durationMs), 100,
                new JavaMockEngineCluster.ClusterStats());
    }

    @AfterEach void close() throws Exception {
        if (prefill != null) prefill.shutdown();
        if (scheduler != null) {
            scheduler.shutdownNow();
            scheduler.awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    private EngineRpcService.TaskInfoPB await(long id, boolean finished) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        do {
            var status = workerStatus(prefill, 0);
            var tasks = finished ? status.getFinishedTaskListList() : status.getRunningTaskInfoList();
            for (var task : tasks) if (task.getRequestId() == id) return task;
            Thread.sleep(5);
        } while (System.nanoTime() < deadline);
        throw new AssertionError("missing " + id + ", finished=" + finished);
    }

    @Test void coldAndWarmCompletionsReportMeasuredPrefixAndStableRequestIdentity() throws Exception {
        start("50");
        enqueueAndFetch(prefill, batch(10, slot(0, inputWithBlockKeys(11, 3072, List.of(91L, 92L)))));
        var cold = await(11, true);
        assertTrue(cold.getPrefixLengthValid(), "a resolved cache miss is a valid zero");
        assertEquals(0, cold.getPrefixLength());
        assertEquals(0, cold.getHbmLocalMatchTokens());
        assertEquals("11", cold.getRequestIdStr());
        // Completion publication precedes the cache handover in the callback.
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (((Number) prefill.getMetricsSnapshot().get("cache_keys")).longValue() != 2
                && System.nanoTime() < deadline) Thread.sleep(5);
        enqueueAndFetch(prefill, batch(20, slot(0, inputWithBlockKeys(21, 3072, List.of(91L, 92L)))));
        var warm = EngineRpcService.TaskInfoPB.parseFrom(await(21, true).toByteArray());
        assertTrue(warm.getPrefixLengthValid());
        assertEquals(2048, warm.getPrefixLength());
        assertEquals(2048, warm.getHbmLocalMatchTokens());
        assertEquals(0, warm.getRemoteKvAddedMatchTokens());
        assertEquals("21", warm.getRequestIdStr());
        assertFalse(warm.hasCompletedPrefillTokens(), "unmodeled progress remains absent");
        assertFalse(warm.hasRemainingPrefillTokens());
    }

    @Test void queuedCancellationDoesNotFabricatePreparedHitFeedback() throws Exception {
        start("5000");
        enqueueAndFetch(prefill, batch(30, slot(0, input(31, 8))));
        var running = await(31, false);
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_RUNNING, running.getPhase());
        assertTrue(running.getPrefixLengthValid());
        enqueueAndFetch(prefill, batch(40, slot(0, input(41, 8))));
        var queued = await(41, false);
        assertEquals(EngineRpcService.TaskPhase.TASK_PHASE_RECEIVED, queued.getPhase());
        assertFalse(queued.getPrefixLengthValid());
        prefill.cancel(41);
        var canceled = await(41, true);
        assertFalse(canceled.getPrefixLengthValid());
        assertEquals("41", canceled.getRequestIdStr());
        assertNotEquals(0, canceled.getErrorInfo().getErrorCode());
    }

    @Test void runningCancellationPreservesResolvedFeedbackInsteadOfLosingIt() throws Exception {
        start("5000");
        enqueueAndFetch(prefill, batch(50, slot(0, input(51, 8))));
        assertTrue(await(51, false).getPrefixLengthValid());
        prefill.cancel(51);
        var canceled = await(51, true);
        assertTrue(canceled.getPrefixLengthValid());
        assertEquals(8, canceled.getInputLength());
        assertEquals("51", canceled.getRequestIdStr());
    }
}
