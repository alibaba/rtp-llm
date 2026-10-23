package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.Consumer;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** Diagnostic rows retain their wire schema and lifetimes independently of live request state. */
class RequestDiagnosticsTest {
    @TempDir Path directory;

    private static void invoke(Object service, String method, Class<?>[] types, Object... args) throws Exception {
        var target = service.getClass().getDeclaredMethod(method, types);
        target.setAccessible(true);
        target.invoke(service, args);
    }

    private static void start(Object service, long id, long batch) throws Exception {
        invoke(service, "recordLifecycleStart", new Class<?>[]{long.class, long.class, String.class}, id, batch, "enqueue_batch");
    }

    private static void stamp(Object service, String method, long id) throws Exception {
        invoke(service, method, new Class<?>[]{long.class}, id);
    }

    @Test
    void duplicateAdmissionAndLateUpdatesCannotRewriteTerminalHistory() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 62100, 1, 0)) {
            var service = cluster.prefill(0);
            start(service, 1, 81);
            var original = service.getRequestLifecycleSnapshot().get("1");
            assertEquals(List.of("rid", "method", "batch_id", "arrived_ms", "running_ms", "end_ms", "end_state"),
                    new ArrayList<>(original.keySet()));
            start(service, 1, 82);
            assertEquals(original, service.getRequestLifecycleSnapshot().get("1"));
            invoke(service, "recordLifecycleEnd", new Class<?>[]{long.class, String.class}, 1L, "failed");
            var terminal = service.getRequestLifecycleSnapshot().get("1");
            invoke(service, "recordLifecycleEnd", new Class<?>[]{long.class, boolean.class}, 1L, false);
            stamp(service, "recordLifecycleRunning", 1);
            assertEquals(terminal, service.getRequestLifecycleSnapshot().get("1"));
            terminal.put("end_state", "mutated snapshot");
            assertEquals("failed", service.getRequestLifecycleSnapshot().get("1").get("end_state"));
            start(service, 1, 83);
            var retry = service.getRequestLifecycleSnapshot().get("1");
            assertEquals(83L, retry.get("batch_id"));
            assertEquals("running", retry.get("end_state"));
            assertEquals(0L, retry.get("end_ms"));
        }
    }

    @Test
    void lifecycleHistoryKeepsOriginalInsertionOrderAndBound() throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "10"), 62100, 1, 0)) {
            var service = cluster.prefill(0);
            for (long id = 1; id <= 10_000; id++) start(service, id, id);
            start(service, 1, 99); // Duplicate admission must not refresh retention order.
            start(service, 10_001, 10_001);
            var history = service.getRequestLifecycleSnapshot();
            assertEquals(10_000, history.size());
            assertFalse(history.containsKey("1"));
            assertEquals("2", history.keySet().iterator().next());
            assertTrue(history.containsKey("10001"));
        }
    }

    private void writeEvent(JavaMockEngineCluster.FastRpcService service, MockPerformanceModel model, long id) throws Exception {
        var cache = service.getClass().getDeclaredField("cache");
        cache.setAccessible(true);
        var shape = model.shape(input(id, 16), (MockLruBlockCache) cache.get(service));
        invoke(service, "writePrefillDoneEvent", new Class<?>[]{MockPerformanceModel.RequestShape.class,
                long.class, long.class, long.class, long.class, int.class, boolean.class},
                shape, id, 7L, System.currentTimeMillis(), 1L, 1, false);
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void firstStampsWinAndTerminalConsumesBothIncludingMetricsOnlyArrival(boolean metricsOnlyArrival) throws Exception {
        var model = performanceModel(directory, "10");
        var path = directory.resolve("events.jsonl");
        try (var cluster = MockEngineTestCluster.create(model, 62100, 1, 0);
             var log = JavaMockEngineCluster.EngineEventLog.open(path.toString())) {
            var service = cluster.prefill(0);
            if (metricsOnlyArrival) {
                var reporter = service.getClass().getDeclaredField("eventMetricReporter");
                reporter.setAccessible(true);
                reporter.set(service, (Consumer<Map<String, Number>>) ignored -> {});
            } else service.setEngineEventLog(log);
            stamp(service, "recordEventArrival", 1);
            long arrivalUpper = System.currentTimeMillis();
            service.setEngineEventLog(log);
            stamp(service, "recordEventStart", 1);
            long startUpper = System.currentTimeMillis();
            Thread.sleep(5);
            stamp(service, "recordEventArrival", 1);
            stamp(service, "recordEventStart", 1);
            writeEvent(service, model, 1);
            writeEvent(service, model, 1);
            var rows = Files.readAllLines(path);
            var mapper = new ObjectMapper();
            var first = mapper.readTree(rows.get(0));
            assertTrue(first.get("engine_arrival_ms").asLong() > 0);
            assertTrue(first.get("engine_arrival_ms").asLong() <= arrivalUpper);
            assertTrue(first.get("prefill_start_ms").asLong() > 0);
            assertTrue(first.get("prefill_start_ms").asLong() <= startUpper);
            var consumed = mapper.readTree(rows.get(1));
            assertEquals(0L, consumed.get("engine_arrival_ms").asLong());
            assertEquals(0L, consumed.get("prefill_start_ms").asLong());
            assertEquals(0L, consumed.get("ttft_ms").asLong());
        }
    }

    @Test
    void housekeepingDropsOrphanStampPairAndRetainsLiveOutputStamps() throws Exception {
        var model = performanceModel(directory, "10");
        var path = directory.resolve("events.jsonl");
        try (var cluster = MockEngineTestCluster.create(model, 62100, 1, 0);
             var log = JavaMockEngineCluster.EngineEventLog.open(path.toString())) {
            var service = cluster.prefill(0);
            service.setEngineEventLog(log);
            for (long id : List.of(1L, 2L)) {
                stamp(service, "recordEventArrival", id);
                stamp(service, "recordEventStart", id);
            }
            service.requests.output(2, new LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>());
            service.periodicCleanup();
            writeEvent(service, model, 1);
            writeEvent(service, model, 2);
            var rows = Files.readAllLines(path);
            var mapper = new ObjectMapper();
            assertEquals(0L, mapper.readTree(rows.get(0)).get("engine_arrival_ms").asLong());
            assertEquals(0L, mapper.readTree(rows.get(0)).get("prefill_start_ms").asLong());
            assertTrue(mapper.readTree(rows.get(1)).get("engine_arrival_ms").asLong() > 0);
            assertTrue(mapper.readTree(rows.get(1)).get("prefill_start_ms").asLong() > 0);
        }
    }
}
