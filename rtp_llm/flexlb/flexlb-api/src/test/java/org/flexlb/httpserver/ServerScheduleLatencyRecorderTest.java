package org.flexlb.httpserver;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ServerScheduleLatencyRecorderTest {

    @Test
    void recordsServerTotalStagesAndRates() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();
        BalanceContext context = new BalanceContext();
        context.setGrpcEntryNanos(end - TimeUnit.MILLISECONDS.toNanos(20));
        context.setServiceStartNanos(end - TimeUnit.MILLISECONDS.toNanos(18));
        context.setRouteSubmittedNanos(end - TimeUnit.MILLISECONDS.toNanos(15));
        context.setBatchDispatchedNanos(end - TimeUnit.MILLISECONDS.toNanos(10));
        context.setAckAtNanos(end - TimeUnit.MILLISECONDS.toNanos(2));

        recorder.recordArrival(end - TimeUnit.SECONDS.toNanos(1));
        recorder.recordArrival(end);
        recorder.recordCompletion(context, end);

        Map<String, Object> snapshot = recorder.snapshot();
        assertEquals(2L, snapshot.get("arrival_count"));
        assertEquals(1.0, (double) snapshot.get("arrival_qps"), 0.001);
        assertLatency(snapshot, "server_total_ms", 20L);
        assertLatency(snapshot, "grpc_queue_ms", 2L);
        assertLatency(snapshot, "route_submit_ms", 3L);
        assertLatency(snapshot, "batch_wait_ms", 5L);
        assertLatency(snapshot, "dispatch_ack_ms", 8L);
        assertLatency(snapshot, "ack_response_ms", 2L);
    }

    @Test
    void bucketsBatchWaitByPriorityAndKeepsGlobalAggregate() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();

        // Budget-carried priorities (Auto-TPM path)
        recorder.recordCompletion(contextWithBatchWait(end, 5, budget(40)), end);
        recorder.recordCompletion(contextWithBatchWait(end, 10, budget(50)), end);
        // Legacy path: priority carried on the request only (budget == null)
        BalanceContext legacy = contextWithBatchWait(end, 20, null);
        Request request = new Request();
        request.setPriority(70);
        legacy.setRequest(request);
        recorder.recordCompletion(legacy, end);

        Map<String, Object> snapshot = recorder.snapshot();
        // Global aggregate contains all three samples (backward-compatible format)
        Map<?, ?> global = (Map<?, ?>) snapshot.get("batch_wait_ms");
        assertEquals(3L, global.get("count"));
        assertEquals(10L, global.get("p50"));
        assertEquals(20L, global.get("p99"));

        Map<?, ?> byPriority = (Map<?, ?>) snapshot.get("batch_wait_ms_by_priority");
        assertEquals(3, byPriority.size());
        assertPriorityBucket(byPriority, 40, 1L, 5L);
        assertPriorityBucket(byPriority, 50, 1L, 10L);
        assertPriorityBucket(byPriority, 70, 1L, 20L);
    }

    @Test
    void fallsBackToPriorityZeroWhenRequestCarriesNoPriority() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();

        // Neither budget nor request present
        recorder.recordCompletion(contextWithBatchWait(end, 5, null), end);
        // Request present but priority unset (proto3 default 0)
        BalanceContext unset = contextWithBatchWait(end, 7, null);
        unset.setRequest(new Request());
        recorder.recordCompletion(unset, end);

        Map<?, ?> byPriority = (Map<?, ?>) recorder.snapshot().get("batch_wait_ms_by_priority");
        assertEquals(1, byPriority.size());
        assertPriorityBucket(byPriority, 0, 2L, 5L);
    }

    @Test
    void logSuffixListsOnlyPrioritiesWithDataInAscendingOrder() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        assertEquals("", recorder.batchWaitPriorityLogSuffix());

        long end = System.nanoTime();
        // Record higher priority first to verify ascending output order
        recorder.recordCompletion(contextWithBatchWait(end, 10, budget(50)), end);
        recorder.recordCompletion(contextWithBatchWait(end, 5, budget(40)), end);

        assertEquals(" batch_wait_p95_prio40_ms=5 batch_wait_p95_prio50_ms=10",
                recorder.batchWaitPriorityLogSuffix());
    }

    @Test
    void resetStartsANewMeasurementWindow() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();
        recorder.recordArrival(end);
        recorder.recordCompletion(contextWithBatchWait(end, 5, budget(40)), end);
        recorder.reset();

        Map<String, Object> snapshot = recorder.snapshot();
        assertEquals(0L, snapshot.get("arrival_count"));
        assertTrue(((Map<?, ?>) snapshot.get("server_total_ms")).containsKey("p99"));
        assertTrue(((Map<?, ?>) snapshot.get("batch_wait_ms_by_priority")).isEmpty());
        assertEquals("", recorder.batchWaitPriorityLogSuffix());
    }

    private static ScheduleBudget budget(int priority) {
        long now = System.currentTimeMillis();
        return ScheduleBudget.forDeadline(priority, now, now + 1000);
    }

    private static BalanceContext contextWithBatchWait(long end, long batchWaitMs, ScheduleBudget budget) {
        BalanceContext context = new BalanceContext();
        context.setGrpcEntryNanos(end - TimeUnit.MILLISECONDS.toNanos(batchWaitMs + 10));
        context.setServiceStartNanos(end - TimeUnit.MILLISECONDS.toNanos(batchWaitMs + 8));
        context.setRouteSubmittedNanos(end - TimeUnit.MILLISECONDS.toNanos(batchWaitMs + 5));
        context.setBatchDispatchedNanos(end - TimeUnit.MILLISECONDS.toNanos(5));
        context.setAckAtNanos(end - TimeUnit.MILLISECONDS.toNanos(2));
        context.setBudget(budget);
        return context;
    }

    private static void assertPriorityBucket(Map<?, ?> byPriority, int priority,
                                             long expectedCount, long expectedP50) {
        Map<?, ?> bucket = (Map<?, ?>) byPriority.get(priority);
        assertEquals(expectedCount, bucket.get("count"),
                "count mismatch for priority " + priority);
        assertEquals(expectedP50, bucket.get("p50"),
                "p50 mismatch for priority " + priority);
        assertTrue(bucket.containsKey("p95"));
        assertTrue(bucket.containsKey("p99"));
        assertTrue(bucket.containsKey("mean"));
    }

    private static void assertLatency(Map<String, Object> snapshot, String name, long expectedMs) {
        Map<?, ?> latency = (Map<?, ?>) snapshot.get(name);
        assertEquals(1L, latency.get("count"));
        assertEquals(expectedMs, latency.get("p50"));
        assertEquals(expectedMs, latency.get("p99"));
    }
}
