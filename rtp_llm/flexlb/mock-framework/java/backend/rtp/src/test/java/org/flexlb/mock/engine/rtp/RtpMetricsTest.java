package org.flexlb.mock.engine.rtp;

import static org.junit.jupiter.api.Assertions.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class RtpMetricsTest {
    private static final RtpMetrics.State IDLE = new RtpMetrics.State(0, 0, 0, 0, 100, 100, 100, 0, 0, 0, 1);

    static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }

    @Test void exposesZeroCountersAndCurrentStateWithoutResetOnScrape() {
        var state = new AtomicReference<>(IDLE);
        var metrics = new RtpMetrics("prefill", state::get);
        assertEquals(0, sample(metrics.scrape(), "rtp_llm_requests_started_total", ""));
        state.set(new RtpMetrics.State(2, 1, 3, 4, 100, 70, 50, 30, 200, 120, 5));
        var before = metrics.scrape();
        assertEquals(2, sample(before, "rtp_llm_scheduler_requests", "state=\"waiting\""));
        assertEquals(70, sample(before, "rtp_llm_kv_tokens", "state=\"available\""));
        assertEquals(3, sample(before, "rtp_llm_scheduler_requests", "state=\"loading\""));
        assertEquals(4, sample(before, "rtp_llm_scheduler_requests", "state=\"ready\""));
        assertEquals(200, sample(before, "rtp_llm_host_kv_tokens", "state=\"total\""));
        assertEquals(120, sample(before, "rtp_llm_host_kv_tokens", "state=\"free\""));
        assertEquals(70, sample(metrics.scrape(), "rtp_llm_kv_tokens", "state=\"available\""));
        state.set(IDLE);
        assertEquals(100, sample(metrics.scrape(), "rtp_llm_kv_tokens", "state=\"available\""));
    }

    @Test void keepsTokenWeightsDurationAndOutcomesSeparate() {
        var metrics = new RtpMetrics("prefill", () -> IDLE);
        metrics.started(); metrics.started();
        metrics.cacheLookup(100, 60); metrics.cacheLookup(10, 0);
        metrics.forwardCompleted(20_000_000, 1000, 100, 40, 1);
        metrics.forwardCompleted(10_000_000, 1000, 10, 10, 1);
        metrics.finished(RtpMetrics.Outcome.success); metrics.finished(RtpMetrics.Outcome.timeout);
        var text = metrics.scrape();
        assertEquals(110, sample(text, "rtp_llm_cache_lookup_tokens_total", ""));
        assertEquals(60, sample(text, "rtp_llm_cache_hit_tokens_total", ""));
        assertEquals(50, sample(text, "rtp_llm_forward_tokens_total", "kind=\"compute\""));
        assertEquals(110, sample(text, "rtp_llm_forward_tokens_total", "kind=\"input\""));
        assertEquals(2, sample(text, "rtp_llm_model_forward_seconds_count", ""));
        assertEquals(.03, sample(text, "rtp_llm_model_forward_seconds_sum", ""), 1e-9);
        assertEquals(10_000, sample(text, "rtp_llm_model_forward_us", ""));
        assertEquals(1, sample(text, "mock_model_forward_planned_us", ""));
        assertFalse(text.contains("node="));
        assertFalse(text.contains("engine="));
        for (String outcome : new String[]{"success", "timeout"})
            assertEquals(1, sample(text, "rtp_llm_framework_requests_total", "outcome=\"" + outcome + "\""));
    }

    @Test void nativeQueueAndKvMetricsKeepRealUnitsAndDoNotInventHostAvailability() {
        var metrics = new RtpMetrics("prefill", () -> new RtpMetrics.State(2, 1, 3, 4, 100, 70, 50, 30, 200, 120, 5));
        var text = metrics.scrape();
        assertEquals(2, sample(text, "rtp_llm_wait_stream_size", ""));
        assertEquals(1, sample(text, "rtp_llm_running_stream_size", ""));
        assertEquals(70, sample(text, "rtp_llm_kv_cache_left_seq", ""));
        assertEquals(20, sample(text, "rtp_llm_kv_cache_total_blocks", "tier=\"DEVICE\""));
        assertEquals(14, sample(text, "rtp_llm_kv_cache_available_blocks", "tier=\"DEVICE\""));
        assertEquals(10, sample(text, "rtp_llm_kv_cache_used_blocks", "tier=\"DEVICE\""));
        assertEquals(30, sample(text, "rtp_llm_kv_cache_used_ratio", "tier=\"DEVICE\""));
        assertEquals(40, sample(text, "rtp_llm_kv_cache_total_blocks", "tier=\"HOST\""));
        assertEquals(24, sample(text, "rtp_llm_kv_cache_free_blocks", "tier=\"HOST\""));
        assertFalse(text.lines().anyMatch(line -> line.startsWith("rtp_llm_kv_cache_available_blocks{") && line.contains("HOST")));
        assertFalse(text.contains("rtp_llm_loading_cache_stream_size"));
        assertFalse(text.contains("mock_engine_"));
        assertThrows(IllegalArgumentException.class, () -> new RtpMetrics.State(0, 0, 0, 0, 100, 71, 50, 0, 0, 0, 5));
    }

    @Test void nativeThroughputSeparatesActiveTimeWallTimeAndTerminalQpsAndClearsIdleWindows() {
        var clock = new java.util.concurrent.atomic.AtomicLong();
        var metrics = new RtpMetrics("prefill", () -> IDLE, clock::get);
        metrics.started(); metrics.started(); metrics.started();
        metrics.forwardCompleted(20_000_000, 1000, 100, 40, 1);
        metrics.forwardCompleted(10_000_000, 1000, 10, 10, 1);
        metrics.finished(RtpMetrics.Outcome.success); metrics.finished(RtpMetrics.Outcome.timeout);
        clock.set(1_000_000_000);
        var text = metrics.scrape();
        assertEquals(50 / .03, sample(text, "rtp_llm_context_tps", ""), 1e-9);
        assertEquals(110 / .03, sample(text, "rtp_llm_context_tps_with_cache", ""), 1e-9);
        assertEquals(50, sample(text, "rtp_llm_context_wall_tps", ""));
        assertEquals(110, sample(text, "rtp_llm_context_wall_tps_with_cache", ""));
        assertEquals(1_000_000, sample(text, "rtp_llm_wall_tps_report_interval_us", ""));
        assertEquals(2, sample(text, "rtp_llm_framework_qps", ""));
        assertEquals(1, sample(text, "rtp_llm_framework_error_qps", ""));
        clock.set(2_000_000_000);
        text = metrics.scrape();
        assertEquals(0, sample(text, "rtp_llm_context_tps", ""));
        assertEquals(0, sample(text, "rtp_llm_framework_qps", ""));
        metrics.forwardCompleted(40_000_000, 1000, 100, 0, 1);
        clock.set(3_000_000_000L);
        text = metrics.scrape();
        assertEquals(0, sample(text, "rtp_llm_context_tps", ""));
        assertEquals(2500, sample(text, "rtp_llm_context_tps_with_cache", ""));
        assertEquals(.03, sample(text, "rtp_llm_context_forward_seconds_total", ""), 1e-9);
        assertEquals(.07, sample(text, "rtp_llm_context_forward_with_cache_seconds_total", ""), 1e-9);
    }

    @Test void decodeOutputCountsCommittedTokensInsteadOfVerificationTokens() {
        var clock = new java.util.concurrent.atomic.AtomicLong();
        var metrics = new RtpMetrics("decode", () -> IDLE, clock::get);
        metrics.forwardCompleted(5_000_000, 1000, 6, 6, 3);
        clock.set(2_000_000_000);
        var text = metrics.scrape();
        assertEquals(1.5, sample(text, "rtp_llm_generate_tps", ""));
        assertEquals(3, sample(text, "rtp_llm_total_tps", ""));
        assertFalse(text.contains("rtp_llm_context_tps"));
    }

    @Test void independentRegistriesDoNotLeakEvenWithIdenticalIdentity() {
        var a = new RtpMetrics("prefill", () -> IDLE);
        var b = new RtpMetrics("prefill", () -> IDLE);
        a.started();
        assertEquals(1, sample(a.scrape(), "rtp_llm_requests_started_total", ""));
        assertEquals(0, sample(b.scrape(), "rtp_llm_requests_started_total", ""));
    }

    @Test void validatesObservationsBeforeChangingAnyCounter() {
        var metrics = new RtpMetrics("prefill", () -> IDLE);
        assertThrows(IllegalArgumentException.class, () -> metrics.cacheLookup(1, 2));
        assertThrows(IllegalArgumentException.class, () -> metrics.forwardCompleted(-1, 1000, 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> metrics.forwardCompleted(1, 1000, 1, 2, 1));
        assertThrows(IllegalArgumentException.class, () -> new RtpMetrics.State(0, 0, 0, 0, 10, 11, 0, 0, 0, 0, 1));
        assertEquals(0, sample(metrics.scrape(), "rtp_llm_cache_lookup_tokens_total", ""));
        assertEquals(0, sample(metrics.scrape(), "rtp_llm_model_forward_seconds_count", ""));
    }

    @Test void concurrentWritersAndScrapesPreserveFinalTotals() throws Exception {
        var metrics = new RtpMetrics("prefill", () -> IDLE);
        try (var threads = Executors.newVirtualThreadPerTaskExecutor()) {
            var writes = new java.util.ArrayList<Future<?>>();
            for (int thread = 0; thread < 8; thread++) writes.add(threads.submit(() -> {
                for (int i = 0; i < 500; i++) {
                    metrics.started(); metrics.cacheLookup(5, 2); metrics.forwardCompleted(1000, 1000, 5, 3, 1);
                    metrics.finished(RtpMetrics.Outcome.success);
                    if (i % 50 == 0) assertFalse(metrics.scrape().isEmpty());
                }
            }));
            for (var write : writes) write.get(5, TimeUnit.SECONDS);
        }
        var text = metrics.scrape();
        assertEquals(4000, sample(text, "rtp_llm_requests_started_total", ""));
        assertEquals(8000, sample(text, "rtp_llm_cache_hit_tokens_total", ""));
        assertEquals(12000, sample(text, "rtp_llm_forward_tokens_total", "kind=\"compute\""));
        assertEquals(4000, sample(text, "rtp_llm_model_forward_seconds_count", ""));
    }
}
