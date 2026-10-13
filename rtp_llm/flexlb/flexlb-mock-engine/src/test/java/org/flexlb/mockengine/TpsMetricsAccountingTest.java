package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.flexlb.mockengine.MockEngineTestSupport.batch;
import static org.flexlb.mockengine.MockEngineTestSupport.enqueue;
import static org.flexlb.mockengine.MockEngineTestSupport.httpGet;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithBlockKeys;
import static org.flexlb.mockengine.MockEngineTestSupport.inputWithDecode;
import static org.flexlb.mockengine.MockEngineTestSupport.performanceModel;
import static org.flexlb.mockengine.MockEngineTestSupport.slot;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Validates the production-caliber TPS series ({@code rtp_llm_context_tps},
 * {@code rtp_llm_context_tps_with_cache}, {@code rtp_llm_generate_tps}) the
 * mock reports on {@code /metrics} in BOTH emission modes (per-engine and
 * role-aggregated).
 *
 * <p>Accounting model under test (see FastRpcService field comments): the
 * series are PURE BOOKKEEPING on completion events — prefill completions add
 * Σ(il−hit) / Σ(il) to the context pair, decode completions add Σ(ol) to
 * generate — and every {@code /metrics} scrape drains the accumulators into
 * the just-scraped window (window = scrape interval; the value read is the
 * token sum of exactly that window, tokens/s under the 1s G1 poller).
 *
 * <ul>
 *   <li>Known il/ol/hitTokens completion events are constructed via
 *       block-cache-key reuse (first request parks its keys in the LRU; the
 *       second identical-key request hits them, fixing hitTokens exactly)</li>
 *   <li>Per-engine mode: prefill engine carries the context pair (compute
 *       excludes hit tokens, with_cache includes them), decode engine the
 *       generate series; a second scrape with no events in between reads 0
 *       (drain semantics — the window, not a lifetime counter)</li>
 *   <li>Role-aggregated mode: prefill bucket = cross-engine context sums,
 *       decode bucket = generate sum</li>
 *   <li>Cancelled completions are EXCLUDED from every numerator (production
 *       semantics: only tokens actually accepted and generated count)</li>
 *   <li>{@code /snapshot} exposes the cumulative {@code hit_tokens_total}
 *       (the cache_saved_tokens source; never drained)</li>
 * </ul>
 */
class TpsMetricsAccountingTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int BASE_PORT = 62900;

    /** {@code metric_name{engine_name=...,role=...,grpc_port="N",...} value} */
    private static final Pattern PER_ENGINE_METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{engine_name=\"[^\"]+\",role=\"[^\"]+\",grpc_port=\"(\\d+)\",engine_ip=\"[^\"]+\"\\}\\s+(\\d+)");

    /** {@code metric_name{role="..."} value} */
    private static final Pattern ROLE_METRIC_PATTERN = Pattern.compile(
            "(\\w+)\\{role=\"(\\w+)\"\\}\\s+(\\d+)");

    @TempDir
    Path tempDir;

    /**
     * Per-engine mode + cache-hit reuse: two identical-key requests fix
     * hitTokens exactly (second request hits all its keys), so
     * context_tps = Σ(il−hit) and context_tps_with_cache = Σ(il) have exact
     * expected values; a follow-up scrape with no events reads 0 (drain
     * semantics); /snapshot carries the cumulative hit_tokens_total.
     */
    @Test
    void perEngineTpsSeriesMatchCompletionAccounting() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "10");

        try (MockEngineTestCluster cluster =
                     MockEngineTestCluster.start(model, BASE_PORT, 1, 1)) {
            JavaMockEngineCluster.FastRpcService prefill = cluster.prefill(0);
            List<Long> keys = List.of(11L, 12L, 13L);

            // Request 1: cold keys — hitTokens = 0, il = 3072 (3 blocks x 1024).
            enqueue(prefill, batch(1000, slot(0,
                    inputWithBlockKeys(100, 3072, keys))));
            cluster.awaitCompleted(1, 5_000);
            cluster.awaitAllInflightZero(2_000);

            // Request 2: same keys — all 3 blocks hit, hitTokens = 3072.
            enqueue(prefill, batch(1001, slot(0,
                    inputWithBlockKeys(101, 3072, keys))));
            cluster.awaitCompleted(2, 5_000);
            cluster.awaitAllInflightZero(2_000);

            String body = httpGet(cluster.controlPort(), "/metrics?per_engine=true");
            Map<String, Map<Integer, Long>> metrics = parsePerEngineMetrics(body);

            Map<Integer, Long> context = metrics.get("rtp_llm_context_tps");
            Map<Integer, Long> contextCache = metrics.get("rtp_llm_context_tps_with_cache");
            Map<Integer, Long> generate = metrics.get("rtp_llm_generate_tps");
            assertNotNull(context, "rtp_llm_context_tps should exist in per-engine /metrics");
            assertNotNull(contextCache,
                    "rtp_llm_context_tps_with_cache should exist in per-engine /metrics");
            assertNotNull(generate, "rtp_llm_generate_tps should exist in per-engine /metrics");

            int prefillPort = prefill.getGrpcPort();
            int decodePort = cluster.decode(0).getGrpcPort();
            // context_tps: (3072 - 0) + (3072 - 3072) = 3072 computed tokens.
            assertEquals(3072L, context.getOrDefault(prefillPort, -1L),
                    "prefill rtp_llm_context_tps must be Σ(il - hit)");
            // with_cache: 3072 + 3072 = 6144 (the reuse shows up here).
            assertEquals(6144L, contextCache.getOrDefault(prefillPort, -1L),
                    "prefill rtp_llm_context_tps_with_cache must be Σ(il)");
            // The prefill engine never generates; the decode engine never prefills.
            assertEquals(0L, generate.getOrDefault(prefillPort, -1L),
                    "prefill engine rtp_llm_generate_tps must stay 0");
            assertEquals(0L, context.getOrDefault(decodePort, -1L),
                    "decode engine rtp_llm_context_tps must stay 0");
            assertEquals(0L, contextCache.getOrDefault(decodePort, -1L),
                    "decode engine rtp_llm_context_tps_with_cache must stay 0");
            assertEquals(0L, generate.getOrDefault(decodePort, -1L),
                    "idle decode engine rtp_llm_generate_tps must be 0");

            // Drain semantics: a second scrape with no events in between
            // reads a fresh (empty) window — the series is per-window
            // accounting, not a lifetime counter.
            String secondBody = httpGet(cluster.controlPort(), "/metrics?per_engine=true");
            Map<String, Map<Integer, Long>> secondMetrics = parsePerEngineMetrics(secondBody);
            assertEquals(0L, secondMetrics.get("rtp_llm_context_tps")
                            .getOrDefault(prefillPort, -1L),
                    "second scrape with no events must read 0 (drain semantics)");

            // Cumulative hit_tokens_total via /snapshot (never drained).
            JsonNode engines = cluster.snapshot();
            long hitTotal = 0;
            for (JsonNode engine : engines) {
                if ("prefill".equals(engine.get("role").asText())) {
                    hitTotal += engine.get("hit_tokens_total").asLong();
                }
            }
            assertEquals(3072L, hitTotal,
                    "prefill hit_tokens_total must be the cumulative Σ(hitTokens)");
        }
    }

    /**
     * Role-aggregated mode over the full P→D pipeline: five requests with
     * known il/ol — prefill bucket carries the context pair, decode bucket
     * the generate series (Σ output tokens, not request count).
     */
    @Test
    void roleAggregatedTpsSeriesMatchCompletionAccounting() throws Exception {
        MockPerformanceModel model = performanceModel(tempDir, "10");

        try (MockEngineTestCluster cluster =
                     MockEngineTestCluster.start(model, BASE_PORT + 20, 1, 1)) {
            JavaMockEngineCluster.FastRpcService prefill = cluster.prefill(0);
            JavaMockEngineCluster.FastRpcService decode = cluster.decode(0);

            int n = 5;
            int inputLen = 1000;
            int outputLen = 7;
            EngineRpcService.GenerateInputPB[] inputs =
                    new EngineRpcService.GenerateInputPB[n];
            for (int i = 0; i < n; i++) {
                inputs[i] = inputWithDecode(200 + i, inputLen,
                        decode.getGrpcPort(), outputLen);
            }
            enqueue(prefill, batch(2000, slot(0, inputs)));
            cluster.awaitCompleted(n, 10_000);
            cluster.awaitAllInflightZero(5_000);

            String body = httpGet(cluster.controlPort(), "/metrics");
            Map<String, Map<String, Long>> roleMetrics = parseRoleMetrics(body);

            Map<String, Long> context = roleMetrics.get("rtp_llm_context_tps");
            Map<String, Long> contextCache = roleMetrics.get("rtp_llm_context_tps_with_cache");
            Map<String, Long> generate = roleMetrics.get("rtp_llm_generate_tps");
            assertNotNull(context, "rtp_llm_context_tps{role=...} should exist in /metrics");
            assertNotNull(contextCache,
                    "rtp_llm_context_tps_with_cache{role=...} should exist in /metrics");
            assertNotNull(generate, "rtp_llm_generate_tps{role=...} should exist in /metrics");

            assertEquals((long) n * inputLen, context.getOrDefault("prefill", -1L),
                    "aggregated prefill rtp_llm_context_tps must be Σ(il - hit), no cache hits here");
            assertEquals((long) n * inputLen, contextCache.getOrDefault("prefill", -1L),
                    "aggregated prefill rtp_llm_context_tps_with_cache must be Σ(il)");
            assertEquals(0L, generate.getOrDefault("prefill", -1L),
                    "aggregated prefill rtp_llm_generate_tps must stay 0");
            // Σ(ol) — the MTP-fold accepted tokens, NOT the request count.
            assertEquals((long) n * outputLen, generate.getOrDefault("decode", -1L),
                    "aggregated decode rtp_llm_generate_tps must be Σ(output_len)");
            assertEquals(0L, context.getOrDefault("decode", -1L),
                    "aggregated decode rtp_llm_context_tps must stay 0");
        }
    }

    /**
     * Cancelled completions are excluded from every TPS numerator: request B
     * is cancelled mid-prefill; only request A's tokens appear in the window.
     */
    @Test
    void cancelledCompletionsExcludedFromTpsAccounting() throws Exception {
        // Fixed 800ms prefill: a stable window to cancel inside.
        MockPerformanceModel model = performanceModel(tempDir, "800");

        try (MockEngineTestCluster cluster =
                     MockEngineTestCluster.start(model, BASE_PORT + 40, 1, 1)) {
            JavaMockEngineCluster.FastRpcService prefill = cluster.prefill(0);
            int inputLen = 512;

            // A completes normally; B is cancelled while running.
            enqueue(prefill, batch(3000, slot(0,
                    inputWithBlockKeys(300, inputLen, List.of()))));
            enqueue(prefill, batch(3001, slot(0,
                    inputWithBlockKeys(301, inputLen, List.of()))));
            cluster.awaitInflight(prefill, 2, 5_000);
            prefill.cancel(301L);
            cluster.awaitAllInflightZero(5_000);

            String body = httpGet(cluster.controlPort(), "/metrics");
            Map<String, Map<String, Long>> roleMetrics = parseRoleMetrics(body);

            assertEquals((long) inputLen,
                    roleMetrics.get("rtp_llm_context_tps").getOrDefault("prefill", -1L),
                    "cancelled request's tokens must not enter rtp_llm_context_tps");
            assertEquals((long) inputLen,
                    roleMetrics.get("rtp_llm_context_tps_with_cache").getOrDefault("prefill", -1L),
                    "cancelled request's tokens must not enter rtp_llm_context_tps_with_cache");
        }
    }

    // ────────────────── helpers ──────────────────

    private static Map<String, Map<Integer, Long>> parsePerEngineMetrics(String body) {
        Map<String, Map<Integer, Long>> result = new java.util.HashMap<>();
        Matcher matcher = PER_ENGINE_METRIC_PATTERN.matcher(body);
        while (matcher.find()) {
            result.computeIfAbsent(matcher.group(1), k -> new java.util.HashMap<>())
                    .put(Integer.parseInt(matcher.group(2)), Long.parseLong(matcher.group(3)));
        }
        return result;
    }

    private static Map<String, Map<String, Long>> parseRoleMetrics(String body) {
        Map<String, Map<String, Long>> result = new java.util.HashMap<>();
        Matcher matcher = ROLE_METRIC_PATTERN.matcher(body);
        while (matcher.find()) {
            result.computeIfAbsent(matcher.group(1), k -> new java.util.HashMap<>())
                    .put(matcher.group(2), Long.parseLong(matcher.group(3)));
        }
        return result;
    }
}
