package org.flexlb.api;

import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks scheduling latency stages and measurement-window resets without inventing
 * missing samples.
 */
class ServerScheduleLatencyRecorderTest {

    @Test
    void ignoresCompletionWithoutContext() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();

        recorder.recordCompletion(null, System.nanoTime());

        assertEquals(0L, recorder.snapshot().get("completion_count"));
        assertEquals(0L, ((Map<?, ?>) recorder.snapshot().get("server_total_ms")).get("count"));
    }

    @Test
    void recordsServerTotalStagesAndRates() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();
        RequestContext context = new RequestContext(org.flexlb.mock.TestFlexlbConfigs.create());
        context.recordGrpcArrival(0L, end - TimeUnit.MILLISECONDS.toNanos(20));
        org.springframework.test.util.ReflectionTestUtils.setField(
                context,
                "timing",
                context.getTiming().withStartedNanos(end - TimeUnit.MILLISECONDS.toNanos(18)));
        org.springframework.test.util.ReflectionTestUtils.setField(
                context,
                "timing",
                context.getTiming()
                        .withRouteCommitStartedNanos(end - TimeUnit.MILLISECONDS.toNanos(15)));

        recorder.recordArrival(end - TimeUnit.SECONDS.toNanos(1));
        recorder.recordArrival(end);
        recorder.recordCompletion(context, end);

        Map<String, Object> snapshot = recorder.snapshot();
        assertEquals(2L, snapshot.get("arrival_count"));
        assertEquals(1.0, (double) snapshot.get("arrival_qps"), 0.001);
        assertLatency(snapshot, "server_total_ms", 20L);
        assertLatency(snapshot, "grpc_queue_ms", 2L);
        assertLatency(snapshot, "route_submit_ms", 3L);
    }

    @Test
    void recordsAllResponseLatencies() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();
        RequestContext terminalFirst = contextWithRoute(end, 20L, null);

        recorder.recordCompletion(terminalFirst, end);
        recorder.recordCompletion(contextWithRoute(end, 5L, null), end);

        Map<String, Object> snapshot = recorder.snapshot();
        assertEquals(2L, snapshot.get("completion_count"));
        for (String stage : new String[] {"server_total_ms", "grpc_queue_ms", "route_submit_ms"}) {
            assertEquals(2L, ((Map<?, ?>) snapshot.get(stage)).get("count"), stage);
        }
        assertEquals(
                30L,
                ((Map<?, ?>) snapshot.get("server_total_ms")).get("max"),
                "the slower terminal-first response must remain in the full latency distribution");
    }

    @Test
    void routePublicationRecordsEndToEndLatencyWithoutInventingBatchOrAckStages() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();
        RequestContext route = contextWithRoute(end, 5L, null);

        recorder.recordCompletion(route, end);

        Map<String, Object> snapshot = recorder.snapshot();
        assertEquals(1L, snapshot.get("completion_count"));
        assertLatency(snapshot, "server_total_ms", 15L);
        assertLatency(snapshot, "grpc_queue_ms", 2L);
        assertLatency(snapshot, "route_submit_ms", 3L);
        assertFalse(snapshot.containsKey("ack_response_ms"));
    }

    @Test
    void resetStartsANewMeasurementWindow() {
        ServerScheduleLatencyRecorder recorder = new ServerScheduleLatencyRecorder();
        long end = System.nanoTime();
        recorder.recordArrival(end);
        recorder.recordCompletion(contextWithRoute(end, 5, metadata(40)), end);
        recorder.reset();

        Map<String, Object> snapshot = recorder.snapshot();
        assertEquals(0L, snapshot.get("arrival_count"));
        assertTrue(((Map<?, ?>) snapshot.get("server_total_ms")).containsKey("p99"));
    }

    private static SchedulingMetadata metadata(int priority) {
        long now = System.currentTimeMillis();
        return SchedulingMetadata.explicit(priority, now + 1000);
    }

    private static void assertLatency(Map<String, Object> snapshot, String name, long expectedMs) {
        Map<?, ?> latency = (Map<?, ?>) snapshot.get(name);
        assertEquals(1L, latency.get("count"));
        assertEquals(expectedMs, latency.get("p50"));
        assertEquals(expectedMs, latency.get("p99"));
    }

    private static RequestContext contextWithRoute(
            long end, long routeWaitMs, SchedulingMetadata metadata) {
        var context = new RequestContext(new org.flexlb.config.FlexlbConfig());
        context.recordGrpcArrival(0L, end - TimeUnit.MILLISECONDS.toNanos(routeWaitMs + 10));
        context.setSchedulingMetadata(metadata);
        org.springframework.test.util.ReflectionTestUtils.setField(
                context,
                "timing",
                context.getTiming()
                        .withStartedNanos(end - TimeUnit.MILLISECONDS.toNanos(routeWaitMs + 8)));
        org.springframework.test.util.ReflectionTestUtils.setField(
                context,
                "timing",
                context.getTiming()
                        .withRouteCommitStartedNanos(
                                end - TimeUnit.MILLISECONDS.toNanos(routeWaitMs + 5)));
        return context;
    }
}
