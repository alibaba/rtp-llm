package org.flexlb.mock.engine.scheduler;

import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import org.flexlb.mock.engine.scheduler.RtpMetrics.StepStats;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class RtpMetricsTest {
    private static final RtpMetrics.State IDLE = new RtpMetrics.State(0, 0, 0, 0, 100, 100, 100, 0, 1);

    private static final RtpMetrics.RequestStats REQUEST =
        new RtpMetrics.RequestStats(20_000, 4_000, 2_000, 0, -1, 10, 3, 2, 4);

    static double sample(String text, String name, String label) {
        return text.lines().filter(line -> line.startsWith(name + "{") && line.contains(label))
            .mapToDouble(line -> Double.parseDouble(line.substring(line.lastIndexOf(' ') + 1))).findFirst().orElseThrow();
    }

    @Test void exposesZeroCountersAndCurrentStateWithoutResetOnScrape() {
        var state = new AtomicReference<>(IDLE);
        var metrics = new RtpMetrics("prefill", state::get);
        assertEquals(0, sample(metrics.scrape(), "rtp_llm_requests_started_total", ""));
        state.set(new RtpMetrics.State(2, 1, 3, 4, 100, 70, 50, 30, 5));
        var before = metrics.scrape();
        assertEquals(2, sample(before, "rtp_llm_scheduler_requests", "state=\"waiting\""));
        assertEquals(70, sample(before, "rtp_llm_kv_tokens", "state=\"available\""));
        assertEquals(3, sample(before, "rtp_llm_scheduler_requests", "state=\"loading\""));
        assertEquals(4, sample(before, "rtp_llm_scheduler_requests", "state=\"ready\""));
        assertEquals(70, sample(metrics.scrape(), "rtp_llm_kv_tokens", "state=\"available\""));
        state.set(IDLE);
        assertEquals(100, sample(metrics.scrape(), "rtp_llm_kv_tokens", "state=\"available\""));
    }

    @Test void keepsTokenWeightsDurationAndOutcomesSeparate() {
        var metrics = new RtpMetrics("prefill", () -> IDLE);
        metrics.started(); metrics.started();
        metrics.cacheLookup(100, 60); metrics.cacheLookup(10, 0);
        metrics.stepCompleted(new StepStats(0, 0, 0, 20_000_000, 20_000_000, 20_000_000, 1000, 100, 40, 1, 0));
        metrics.stepCompleted(new StepStats(0, 0, 0, 10_000_000, 10_000_000, 10_000_000, 1000, 10, 10, 1, 0));
        metrics.requestFinished(RtpMetrics.Outcome.success, REQUEST, true);
        metrics.requestFinished(RtpMetrics.Outcome.timeout, REQUEST, true);
        var text = metrics.scrape();
        assertEquals(110, sample(text, "rtp_llm_cache_lookup_tokens_total", ""));
        assertEquals(60, sample(text, "rtp_llm_cache_hit_tokens_total", ""));
        assertEquals(50, sample(text, "rtp_llm_forward_tokens_total", "kind=\"compute\""));
        assertEquals(110, sample(text, "rtp_llm_forward_tokens_total", "kind=\"input\""));
        assertEquals(2, sample(text, "rtp_llm_model_forward_seconds_count", ""));
        assertEquals(.03, sample(text, "rtp_llm_model_forward_seconds_sum", ""), 1e-9);
        assertEquals(15_000, sample(text, "rtp_llm_model_forward_us", ""));
        assertEquals(10_000, sample(text, "mock_model_forward_actual_us", ""));
        assertFalse(metrics.scrape().contains("rtp_llm_model_forward_us{"));
        assertEquals(1, sample(text, "mock_model_forward_planned_us", ""));
        assertFalse(text.contains("node="));
        assertFalse(text.contains("engine="));
        for (String outcome : new String[]{"success", "timeout"})
            assertEquals(1, sample(text, "rtp_llm_framework_requests_total", "outcome=\"" + outcome + "\""));
    }

    @Test void reportsActualReuseAtCompletionWhenEnabled() {
        var prefill = new RtpMetrics("prefill", () -> IDLE);
        prefill.prepared(10, 4, 100);
        assertFalse(prefill.scrape().contains("rtp_llm_kv_cache_reuse_length"));
        prefill.requestFinished(RtpMetrics.Outcome.success, REQUEST, true);
        var completed = prefill.scrape();
        assertEquals(4, sample(completed, "rtp_llm_kv_cache_reuse_length", "role=\"prefill\""));
        assertEquals(10, sample(completed, "rtp_llm_input_token_length", "role=\"prefill\""));

        var decode = new RtpMetrics("decode", () -> IDLE);
        decode.prepared(10, 4, 100);
        decode.requestFinished(RtpMetrics.Outcome.success, REQUEST, true);
        var decoded = decode.scrape();
        assertEquals(4, sample(decoded, "rtp_llm_kv_cache_reuse_length", "role=\"decode\""));
        assertEquals(10, sample(decoded, "rtp_llm_input_token_length", "role=\"decode\""));

        var disabled = new RtpMetrics("decode", () -> IDLE);
        disabled.requestFinished(RtpMetrics.Outcome.success, REQUEST, false);
        assertFalse(disabled.scrape().contains("rtp_llm_kv_cache_reuse_length"));
    }

    @Test void theoryCountsPriorCompleteBlocksAndExpiresOldKeys() {
        var clock = new AtomicLong();
        var metrics = new RtpMetrics("prefill", () -> IDLE, clock::get);
        var hit = "rtp_llm_prefill_worker_recent_cache_key_hit_count";
        var total = "rtp_llm_prefill_worker_recent_cache_key_total_count";
        metrics.recordPrefillCacheKeys(List.of(1L, 2L), 10, 4);
        clock.addAndGet(TimeUnit.MINUTES.toNanos(20));
        var first = metrics.scrape();
        assertEquals(0, sample(first, hit, ""));
        assertEquals(10, sample(first, total, ""));

        metrics.recordPrefillCacheKeys(List.of(1L), 6, 4);
        clock.addAndGet(TimeUnit.MINUTES.toNanos(11));
        assertEquals(4, sample(metrics.scrape(), hit, ""));
        metrics.recordPrefillCacheKeys(List.of(1L, 2L), 10, 4);
        clock.incrementAndGet();
        assertEquals(4, sample(metrics.scrape(), hit, ""));
    }

    @Test void batchAndForwardMetricsFollowWorkAndProductionRole() {
        var p = new RtpMetrics("prefill", () -> IDLE);
        p.stepCompleted(new StepStats(2, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0));
        var prefill = p.scrape();
        assertEquals(2, sample(prefill, "rtp_llm_context_batch_size_when_has_context", ""));
        assertFalse(prefill.contains("rtp_llm_generate_batch_size"));
        assertFalse(prefill.contains("rtp_llm_execute_token_size"));
        p.stepCompleted(new StepStats(2, 3, 12, 0, 0, 0, 0, 10, 10, 0, 0));
        assertEquals(3, sample(p.scrape(), "rtp_llm_generate_batch_size_when_has_context", ""));

        var d = new RtpMetrics("decode", () -> IDLE);
        d.stepCompleted(new StepStats(0, 3, 12, 5_000_000, 5_000_000, 5_000_000, 4_000_000, 6, 6, 3, 0));
        var decode = d.scrape();
        assertEquals(3, sample(decode, "rtp_llm_generate_batch_size", ""));
        assertEquals(5_000, sample(decode, "mock_model_forward_actual_us", ""));
        assertFalse(decode.contains("rtp_llm_model_forward_us"));
        assertFalse(decode.contains("rtp_llm_context_batch_size"));
        assertFalse(decode.contains("_when_has_context"));
    }

    @Test void stepAndTpotSamplesOnlyDescribeCompletedWork() {
        var prefill = new RtpMetrics("prefill", () -> IDLE);
        prefill.stepCompleted(new StepStats(1, 0, 12, 7_000_000, 7_000_000, 0, 0, 0, 0, 0, 0));
        var text = prefill.scrape();
        assertEquals(7000, sample(text, "rtp_llm_step_latency_us", ""));
        assertFalse(text.contains("mock_decode_estimate_tpot_us"));

        var decode = new RtpMetrics("decode", () -> IDLE);
        decode.stepCompleted(new StepStats(0, 1, 12, 6_000_000, 6_000_000, 0, 0, 0, 0, 0, 0));
        assertFalse(decode.scrape().contains("mock_decode_estimate_tpot_us"));
        decode.stepCompleted(new StepStats(0, 2, 12, 9_000_000, 9_000_000, 0, 0, 2, 2, 6, 2));
        text = decode.scrape();
        assertEquals(9000, sample(text, "rtp_llm_step_latency_us", ""));
        assertEquals(3000, sample(text, "mock_decode_estimate_tpot_us", ""));

    }

    @Test void nativeQueueAndKvMetricsKeepRealUnitsAndDoNotInventHostAvailability() {
        var metrics = new RtpMetrics("prefill", () -> new RtpMetrics.State(2, 1, 3, 4, 100, 70, 50, 30, 5));
        var text = metrics.scrape();
        assertEquals(2, sample(text, "rtp_llm_wait_stream_size", ""));
        assertEquals(1, sample(text, "rtp_llm_running_stream_size", ""));
        assertEquals(70, sample(text, "rtp_llm_kv_cache_left_seq", ""));
        assertEquals(20, sample(text, "rtp_llm_kv_cache_total_blocks", "tier=\"DEVICE\""));
        assertEquals(14, sample(text, "rtp_llm_kv_cache_available_blocks", "tier=\"DEVICE\""));
        assertEquals(10, sample(text, "rtp_llm_kv_cache_used_blocks", "tier=\"DEVICE\""));
        assertEquals(6, sample(text, "rtp_llm_kv_cache_active_blocks", "tier=\"DEVICE\""));
        assertEquals(30, sample(text, "rtp_llm_kv_cache_used_ratio", "tier=\"DEVICE\""));
        assertFalse(text.contains("tier=\"HOST\""));
        assertFalse(text.contains("mock_engine_"));
        assertThrows(IllegalArgumentException.class, () -> new RtpMetrics.State(0, 0, 0, 0, 100, 71, 50, 0, 5));
    }

    @Test void nativeThroughputSeparatesActiveTimeWallTimeAndTerminalQpsAndClearsIdleWindows() {
        var clock = new java.util.concurrent.atomic.AtomicLong();
        var metrics = new RtpMetrics("prefill", () -> IDLE, clock::get);
        metrics.started(); metrics.started(); metrics.started();
        metrics.stepCompleted(new StepStats(0, 0, 0, 20_000_000, 20_000_000, 20_000_000, 1000, 100, 40, 1, 0));
        metrics.stepCompleted(new StepStats(0, 0, 0, 10_000_000, 10_000_000, 10_000_000, 1000, 10, 10, 1, 0));
        metrics.requestFinished(RtpMetrics.Outcome.success, REQUEST, true);
        metrics.requestFinished(RtpMetrics.Outcome.timeout, REQUEST, true);
        clock.set(1_000_000_000);
        var text = metrics.scrape();
        assertFalse(text.contains("rtp_llm_generate_tps"));
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
        metrics.stepCompleted(new StepStats(0, 0, 0, 40_000_000, 40_000_000, 40_000_000, 1000, 100, 0, 1, 0));
        clock.set(3_000_000_000L);
        text = metrics.scrape();
        assertFalse(text.contains("rtp_llm_context_tps{"));
        assertEquals(2500, sample(text, "rtp_llm_context_tps_with_cache", ""));
        assertEquals(.03, sample(text, "rtp_llm_context_forward_seconds_total", ""), 1e-9);
        assertEquals(.07, sample(text, "rtp_llm_context_forward_with_cache_seconds_total", ""), 1e-9);
    }

    @Test void requestFinishRecordsTimingsOnlyForNativeStreamOutcomes() {
        var metrics = new RtpMetrics("decode", () -> IDLE);
        metrics.requestFinished(RtpMetrics.Outcome.error, REQUEST, true);
        var error = metrics.scrape();
        assertEquals(1, sample(error, "rtp_llm_framework_requests_total", "outcome=\"error\""));
        assertFalse(error.contains("rtp_llm_latency_us{"));

        metrics.requestFinished(RtpMetrics.Outcome.cancelled, REQUEST, true);
        var cancelled = metrics.scrape();
        assertEquals(1, sample(cancelled, "rtp_llm_framework_requests_total", "outcome=\"cancelled\""));
        assertEquals(20, sample(cancelled, "rtp_llm_latency_us", ""));
        assertEquals(4, sample(cancelled, "rtp_llm_wait_latency_us", ""));
        assertEquals(2, sample(cancelled, "rtp_llm_stream_enqueue_to_canrun_us", ""));
        assertEquals(3, sample(cancelled, "rtp_llm_output_token_length", ""));
        assertFalse(cancelled.contains("rtp_llm_first_token_latency_us{"));
        assertFalse(cancelled.contains("rtp_llm_stream_canrun_to_running_us{"));
    }

    @Test void decodeThroughputUsesActiveTimeAndPreservesCommittedOutputCounter() {
        var clock = new java.util.concurrent.atomic.AtomicLong();
        var metrics = new RtpMetrics("decode", () -> IDLE, clock::get);
        metrics.stepCompleted(new StepStats(0, 1, 12, 8_000_000, 8_000_000, 5_000_000, 1000, 6, 6, 3, 1));
        metrics.stepCompleted(new StepStats(0, 0, 0, 50_000_000, 50_000_000, 40_000_000, 1000, 4, 4, 0, 0));
        clock.set(2_000_000_000);
        var text = metrics.scrape();
        assertEquals(375, sample(text, "rtp_llm_generate_tps", ""));
        assertEquals(375, sample(text, "rtp_llm_total_tps", ""));
        assertEquals(3, sample(text, "rtp_llm_forward_tokens_total", "kind=\"output\""));
        assertFalse(text.contains("rtp_llm_context_tps"));

        clock.set(3_000_000_000L);
        assertEquals(0, sample(metrics.scrape(), "rtp_llm_generate_tps", ""));
        metrics.stepCompleted(new StepStats(0, 0, 0, 10_000_000, 10_000_000, 10_000_000, 1000, 4, 4, 2, 0));
        clock.set(4_000_000_000L);
        assertEquals(200, sample(metrics.scrape(), "rtp_llm_generate_tps", ""));
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
        assertThrows(IllegalArgumentException.class, () -> metrics.stepCompleted(new StepStats(0, 0, 0, -1, -1, -1, 1000, 1, 1, 1, 0)));
        assertThrows(IllegalArgumentException.class, () -> metrics.stepCompleted(new StepStats(0, 0, 0, 1, 1, 1, 1000, 1, 2, 1, 0)));
        assertThrows(IllegalArgumentException.class, () -> new RtpMetrics.State(0, 0, 0, 0, 10, 11, 0, 0, 1));
        assertEquals(0, sample(metrics.scrape(), "rtp_llm_cache_lookup_tokens_total", ""));
        assertEquals(0, sample(metrics.scrape(), "rtp_llm_model_forward_seconds_count", ""));
    }

    @Test void concurrentWritersAndScrapesPreserveFinalTotals() throws Exception {
        var metrics = new RtpMetrics("prefill", () -> IDLE);
        try (var threads = Executors.newVirtualThreadPerTaskExecutor()) {
            var writes = new java.util.ArrayList<Future<?>>();
            for (int thread = 0; thread < 8; thread++) writes.add(threads.submit(() -> {
                for (int i = 0; i < 500; i++) {
                    metrics.started(); metrics.cacheLookup(5, 2); metrics.stepCompleted(new StepStats(0, 0, 0, 1000, 1000, 1000, 1000, 5, 3, 1, 0));
                    metrics.requestFinished(RtpMetrics.Outcome.success, REQUEST, true);
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
