package org.flexlb.mock.metrics;

import static org.junit.jupiter.api.Assertions.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class EngineMetricsTest {
    private static final EngineMetrics.State IDLE = new EngineMetrics.State(0, 0, 100, 100, 100, 0);

    static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }

    @Test void exposesZeroCountersAndCurrentStateWithoutResetOnScrape() {
        var state = new AtomicReference<>(IDLE);
        var metrics = new EngineMetrics("rtp", "prefill", "p-0", state::get);
        assertEquals(0, sample(metrics.scrape(), "mock_engine_requests_started_total", ""));
        state.set(new EngineMetrics.State(2, 1, 100, 70, 50, 30));
        var before = metrics.scrape();
        assertEquals(2, sample(before, "mock_engine_requests", "state=\"waiting\""));
        assertEquals(70, sample(before, "mock_engine_kv_tokens", "state=\"available\""));
        assertEquals(before, metrics.scrape());
        state.set(IDLE);
        assertEquals(100, sample(metrics.scrape(), "mock_engine_kv_tokens", "state=\"available\""));
    }

    @Test void keepsTokenWeightsDurationAndOutcomesSeparate() {
        var metrics = new EngineMetrics("rtp", "prefill", "p-0", () -> IDLE);
        metrics.started(); metrics.started();
        metrics.cacheLookup(100, 60); metrics.cacheLookup(10, 0);
        metrics.forwardCompleted(20_000_000, 100, 40, 1);
        metrics.forwardCompleted(10_000_000, 10, 10, 1);
        metrics.finished(EngineMetrics.Outcome.success); metrics.finished(EngineMetrics.Outcome.timeout);
        var text = metrics.scrape();
        assertEquals(110, sample(text, "mock_engine_cache_lookup_tokens_total", ""));
        assertEquals(60, sample(text, "mock_engine_cache_hit_tokens_total", ""));
        assertEquals(50, sample(text, "mock_engine_forward_tokens_total", "kind=\"compute\""));
        assertEquals(110, sample(text, "mock_engine_forward_tokens_total", "kind=\"input\""));
        assertEquals(2, sample(text, "mock_engine_forward_seconds_count", ""));
        assertEquals(.03, sample(text, "mock_engine_forward_seconds_sum", ""), 1e-9);
        for (String outcome : new String[]{"success", "timeout"})
            assertEquals(1, sample(text, "mock_engine_requests_finished_total", "outcome=\"" + outcome + "\""));
    }

    @Test void independentRegistriesDoNotLeakEvenWithIdenticalIdentity() {
        var a = new EngineMetrics("rtp", "prefill", "p-0", () -> IDLE);
        var b = new EngineMetrics("rtp", "prefill", "p-0", () -> IDLE);
        a.started();
        assertEquals(1, sample(a.scrape(), "mock_engine_requests_started_total", ""));
        assertEquals(0, sample(b.scrape(), "mock_engine_requests_started_total", ""));
    }

    @Test void validatesObservationsBeforeChangingAnyCounter() {
        var metrics = new EngineMetrics("rtp", "prefill", "p-0", () -> IDLE);
        assertThrows(IllegalArgumentException.class, () -> metrics.cacheLookup(1, 2));
        assertThrows(IllegalArgumentException.class, () -> metrics.forwardCompleted(-1, 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> metrics.forwardCompleted(1, 1, 2, 1));
        assertThrows(IllegalArgumentException.class, () -> new EngineMetrics.State(0, 0, 10, 11, 0, 0));
        assertEquals(0, sample(metrics.scrape(), "mock_engine_cache_lookup_tokens_total", ""));
        assertEquals(0, sample(metrics.scrape(), "mock_engine_forward_seconds_count", ""));
    }

    @Test void concurrentWritersAndScrapesPreserveFinalTotals() throws Exception {
        var metrics = new EngineMetrics("vllm", "prefill", "p-0", () -> IDLE);
        try (var threads = Executors.newVirtualThreadPerTaskExecutor()) {
            var writes = new java.util.ArrayList<Future<?>>();
            for (int thread = 0; thread < 8; thread++) writes.add(threads.submit(() -> {
                for (int i = 0; i < 500; i++) {
                    metrics.started(); metrics.cacheLookup(5, 2); metrics.forwardCompleted(1000, 5, 3, 1);
                    metrics.finished(EngineMetrics.Outcome.success);
                    if (i % 50 == 0) assertFalse(metrics.scrape().isEmpty());
                }
            }));
            for (var write : writes) write.get(5, TimeUnit.SECONDS);
        }
        var text = metrics.scrape();
        assertEquals(4000, sample(text, "mock_engine_requests_started_total", ""));
        assertEquals(8000, sample(text, "mock_engine_cache_hit_tokens_total", ""));
        assertEquals(12000, sample(text, "mock_engine_forward_tokens_total", "kind=\"compute\""));
        assertEquals(4000, sample(text, "mock_engine_forward_seconds_count", ""));
    }
}
