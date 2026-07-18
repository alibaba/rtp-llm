package org.flexlb.httpserver;

import org.flexlb.dao.BalanceContext;
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
    void resetStartsANewMeasurementWindow() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        recorder.recordArrival(System.nanoTime());
        recorder.reset();

        assertEquals(0L, recorder.snapshot().get("arrival_count"));
        assertTrue(((Map<?, ?>) recorder.snapshot().get("server_total_ms")).containsKey("p99"));
    }

    private static void assertLatency(Map<String, Object> snapshot, String name, long expectedMs) {
        Map<?, ?> latency = (Map<?, ?>) snapshot.get(name);
        assertEquals(1L, latency.get("count"));
        assertEquals(expectedMs, latency.get("p50"));
        assertEquals(expectedMs, latency.get("p99"));
    }
}
