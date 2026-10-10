package org.flexlb.mockengine;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Shared metric identity, role, unit and HTTP projection. Sink registration remains adapter-specific. */
final class MockMetricContract {
    enum Type { COUNTER, GAUGE, HISTOGRAM }
    enum Unit { COUNT, REQUESTS_PER_SECOND, TOKENS, BLOCKS, MICROSECONDS, MILLISECONDS, TOKENS_PER_SECOND, PERCENT, RATIO }
    enum Role { PREFILL, DECODE, BOTH }
    enum Aggregation { SUM, WEIGHTED_MEAN, HISTOGRAM, NONE }
    enum Sampling { SNAPSHOT, EVENT, SCHEDULER }
    record Metric(String name, String field, String help, Type type, Unit unit,
                  Role role, Aggregation aggregation, Sampling sampling) {
        boolean belongsTo(String roleName) {
            return role == Role.BOTH || role.name().equalsIgnoreCase(roleName.replace("ROLE_TYPE_", ""));
        }
    }

    static final List<Metric> ALL = List.of(
        metric("mock_context_compute_tokens_total", "context_compute_tokens_total", "cumulative computed input tokens", Type.COUNTER, Unit.TOKENS, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_context_tokens_total", "context_tokens_total", "cumulative input tokens including hits", Type.COUNTER, Unit.TOKENS, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_hit_tokens_total", "hit_tokens_total", "cache hit tokens of completed prefill requests", Type.COUNTER, Unit.TOKENS, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_context_requests_total", "context_requests_total", "completed prefill requests", Type.COUNTER, Unit.COUNT, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_admission_open", "admission_open", "whether new work RPCs may enter", Type.GAUGE, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_admitted_rpcs_total", "admitted_rpcs_total", "work RPCs admitted at entry before removal", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_rejected_rpcs_total", "rejected_rpcs_total", "work RPCs rejected by removal admission gate", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_prefill_batch_size", "prefill_batch_size_buckets", "executed prefill batch size in requests", Type.HISTOGRAM, Unit.COUNT, Role.PREFILL, Aggregation.HISTOGRAM, Sampling.SNAPSHOT),
        metric("mock_generate_tokens_total", "generate_tokens_total", "cumulative output tokens of completed requests", Type.COUNTER, Unit.TOKENS, Role.DECODE, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("rtp_llm_running_stream_size", "scheduler_running", "currently executing scheduler streams", Type.GAUGE, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SCHEDULER),
        metric("rtp_llm_wait_stream_size", "waiting", "scheduler waiting streams (excludes pre-GENERATE decode reservations)", Type.GAUGE, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_accepted_total", "accepted", "total accepted requests", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_completed_total", "completed", "total completed requests", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_cache_evictions_total", "cache_evictions", "total cache evictions", Type.COUNTER, Unit.BLOCKS, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_prefill_ms_avg", "prefill_ms_avg", "average prefill execution time in ms", Type.GAUGE, Unit.MILLISECONDS, Role.PREFILL, Aggregation.WEIGHTED_MEAN, Sampling.SNAPSHOT),
        metric("mock_engine_decode_ms_avg", "decode_ms_avg", "average decode execution time in ms", Type.GAUGE, Unit.MILLISECONDS, Role.DECODE, Aggregation.WEIGHTED_MEAN, Sampling.SNAPSHOT),
        metric("rtp_llm_context_tps", "context_tps", "computed context tokens per second of corresponding batch execution", Type.GAUGE, Unit.TOKENS_PER_SECOND, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("rtp_llm_context_tps_with_cache", "context_tps_with_cache", "context tokens including cache hits per second of corresponding batch execution", Type.GAUGE, Unit.TOKENS_PER_SECOND, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("rtp_llm_context_wall_tps", "context_wall_tps", "computed context tokens per elapsed report second", Type.GAUGE, Unit.TOKENS_PER_SECOND, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("rtp_llm_context_wall_tps_with_cache", "context_wall_tps_with_cache", "context tokens including cache hits per elapsed report second", Type.GAUGE, Unit.TOKENS_PER_SECOND, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("rtp_llm_wall_tps_report_interval_us", "wall_tps_report_interval_us", "elapsed prefill reporting window in microseconds", Type.GAUGE, Unit.MICROSECONDS, Role.PREFILL, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_generate_tps", "generate_tps", "executed decode tokens in reporting window (real legacy gauge)", Type.GAUGE, Unit.TOKENS, Role.DECODE, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_decode_wall_tps", "decode_wall_tps", "executed decode tokens per elapsed reporting second", Type.GAUGE, Unit.TOKENS_PER_SECOND, Role.DECODE, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("rtp_llm_kv_cache_pool_total_blocks", "cache_blocks", "total block-pool size in blocks", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("rtp_llm_kv_cache_pool_available_blocks", "available_blocks", "available blocks (free + pure-LRU, held excluded)", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_held_blocks", "held_blocks", "blocks held by in-flight requests", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_referenced_blocks", "referenced_blocks", "cache-key blocks referenced by in-flight requests", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_kv_admission_fails_total", "kv_admission_fails", "total decode KV admission/growth failures, RETRYABLE family (temporarily short; 8211 terminals after the ALLOCATE retry window)", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_lack_mem_rejects_total", "lack_mem_rejects", "total LACK_MEM rejections, PERMANENT family (never fits) + prefill pool 602 surface", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_decode_reuse_blocks_total", "decode_reuse_blocks", "total decode prefix-reuse blocks (own-LRU net-demand deduction)", Type.COUNTER, Unit.BLOCKS, Role.DECODE, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_cache_key_hits_total", "cache_key_hits", "total prefix-matched cache keys at prefill admission", Type.COUNTER, Unit.COUNT, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_engine_cache_keys_requested_total", "cache_keys_requested", "total request block keys observed at prefill admission (empty-bh adds 0)", Type.COUNTER, Unit.COUNT, Role.PREFILL, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_backend_latency_us", null, "Whale engine observation", Type.GAUGE, Unit.MICROSECONDS, Role.DECODE, Aggregation.NONE, Sampling.EVENT),
        metric("mock_backend_ttft_us", null, "Whale engine observation", Type.GAUGE, Unit.MICROSECONDS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("mock_cancelled_requests_total", null, "Whale engine observation", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_decode_reserved_requests", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.DECODE, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_decode_step_tokens_total", "decode_step_tokens_total", "cumulative tokens produced by decode execution steps", Type.COUNTER, Unit.TOKENS, Role.DECODE, Aggregation.SUM, Sampling.SNAPSHOT),
        metric("mock_decode_success_qps", null, "Whale engine observation", Type.GAUGE, Unit.REQUESTS_PER_SECOND, Role.DECODE, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_decode_waiting_requests", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.DECODE, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_kv_available_tokens", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_kv_occupied_tokens", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_kv_total_tokens", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_evicted_blocks_total", null, "Whale engine observation", Type.COUNTER, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_occupancy_ratio", null, "Whale engine observation", Type.GAUGE, Unit.RATIO, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_pending_write_blocks", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_pinned_blocks", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_read_blocks_total", null, "Whale engine observation", Type.COUNTER, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_total_tokens", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_write_rejected_total", null, "Whale engine observation", Type.COUNTER, Unit.COUNT, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_evicted_block_lifetime_ms", null, "Whale engine observation", Type.GAUGE, Unit.MILLISECONDS, Role.BOTH, Aggregation.NONE, Sampling.EVENT),
        metric("mock_prefill_kv_match_ratio", null, "Whale engine observation", Type.GAUGE, Unit.RATIO, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("mock_prefill_admitted_requests", null, "admitted prefill requests including reserved execution batches", Type.GAUGE, Unit.COUNT, Role.PREFILL, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_context_batch_size", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.PREFILL, Aggregation.NONE, Sampling.SCHEDULER),
        metric("rtp_llm_effective_context_length", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_first_token_latency_us", null, "Whale engine observation", Type.GAUGE, Unit.MICROSECONDS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_generate_batch_size", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.DECODE, Aggregation.NONE, Sampling.SCHEDULER),
        metric("rtp_llm_input_token_length", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.BOTH, Aggregation.NONE, Sampling.EVENT),
        metric("mock_cache_direct_evicted_blocks", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.EVENT),
        metric("mock_cache_evicted_entry_age_ms", null, "Whale engine observation", Type.GAUGE, Unit.MILLISECONDS, Role.BOTH, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_kv_cache_hit_rate", null, "one-minute token-weighted cache hit percentage", Type.GAUGE, Unit.PERCENT, Role.PREFILL, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_kv_cache_item_num", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_kv_cache_left_seq", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_allocated_blocks", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_available_blocks", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_total_blocks", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("mock_memory_cache_unavailable_ratio", null, "Whale engine observation", Type.GAUGE, Unit.PERCENT, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_kv_cache_pool_free_blocks", null, "Whale engine observation", Type.GAUGE, Unit.BLOCKS, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_kv_cache_pool_used_ratio", null, "Whale engine observation", Type.GAUGE, Unit.PERCENT, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_kv_cache_reuse_length", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_latency_us", null, "Whale engine observation", Type.GAUGE, Unit.MICROSECONDS, Role.DECODE, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_loading_cache_stream_size", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_model_forward_us", null, "Whale engine observation", Type.GAUGE, Unit.MICROSECONDS, Role.BOTH, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_output_token_length", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.BOTH, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_recent_cache_key_hit_count", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_recent_cache_key_hit_ratio", null, "Whale engine observation", Type.GAUGE, Unit.RATIO, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_recent_cache_key_retained_occurrences", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_recent_cache_key_retained_unique_cache_keys", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_recent_cache_key_time_window_ms", null, "Whale engine observation", Type.GAUGE, Unit.MILLISECONDS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_recent_cache_key_total_count", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_theory_cache_all_hit_ratio", null, "Whale engine observation", Type.GAUGE, Unit.RATIO, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_theory_cache_all_hit_tokens", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_prefill_worker_theory_cache_all_input_tokens", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_remote_running_stream_size", null, "Whale engine observation", Type.GAUGE, Unit.COUNT, Role.BOTH, Aggregation.NONE, Sampling.SNAPSHOT),
        metric("rtp_llm_reuse_length", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_sp_avg_accept_token_num", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.DECODE, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_sp_estimate_tpot_us", null, "Whale engine observation", Type.GAUGE, Unit.MICROSECONDS, Role.DECODE, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_sp_step_latency_us", null, "Whale engine observation", Type.GAUGE, Unit.MICROSECONDS, Role.DECODE, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_sp_total_accepted_token_num", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.DECODE, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_kv_cache_device_reuse_length", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT),
        metric("rtp_llm_kv_cache_host_reuse_length", null, "Whale engine observation", Type.GAUGE, Unit.TOKENS, Role.PREFILL, Aggregation.NONE, Sampling.EVENT)
    );
    static final List<Metric> HTTP = ALL.stream().filter(metric -> metric.field() != null).toList();
    private static final Map<String, Metric> BY_NAME = index();

    private static Metric metric(String name, String field, String help, Type type, Unit unit,
                                 Role role, Aggregation aggregation, Sampling sampling) {
        return new Metric(name, field, help, type, unit, role, aggregation, sampling);
    }
    private static Map<String, Metric> index() {
        Map<String, Metric> result = new LinkedHashMap<>();
        for (Metric metric : ALL)
            if (result.put(metric.name(), metric) != null)
                throw new IllegalStateException("Duplicate metric: " + metric.name());
        return Map.copyOf(result);
    }
    static Metric require(String name) {
        Metric metric = BY_NAME.get(name);
        if (metric == null) throw new IllegalArgumentException("Unknown mock metric: " + name);
        return metric;
    }
    private MockMetricContract() {}
}
