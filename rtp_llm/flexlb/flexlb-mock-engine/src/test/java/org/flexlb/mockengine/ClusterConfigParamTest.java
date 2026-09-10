package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Phase 2 tests for {@link JavaMockEngineCluster.Config} validation
 * relaxation (single-role clusters) and the new launcher parameters
 * (--total-kv-tokens, --block-size, --decode-max-concurrency).
 */
class ClusterConfigParamTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    private String[] baseArgs() {
        return new String[]{
                "--endpoint-file", tempDir.resolve("endpoint.json").toString(),
                "--performance", tempDir.resolve("performance.json").toString(),
                "--master-config", tempDir.resolve("master.json").toString(),
        };
    }

    private static String[] with(String[] base, String... extra) {
        String[] args = new String[base.length + extra.length];
        System.arraycopy(base, 0, args, 0, base.length);
        System.arraycopy(extra, 0, args, base.length, extra.length);
        return args;
    }

    @Test
    void fetchIsRequiredByDefaultAndShortcutIsExplicit() {
        var strict = JavaMockEngineCluster.Config.parse(baseArgs());
        assertEquals(false, strict.autoFetch);
        assertEquals(600_000L, strict.fetchAttachTimeoutMs);
        var automatic = JavaMockEngineCluster.Config.parse(with(baseArgs(),
                "--auto-fetch", "true", "--fetch-attach-timeout-ms", "25"));
        assertTrue(automatic.autoFetch);
        assertEquals(25L, automatic.fetchAttachTimeoutMs);
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--fetch-attach-timeout-ms", "0")));
    }

    // ──────────── Single-role validation relaxation ────────────

    @Test
    void parseAllowsPrefillOnlyCluster() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--n-prefill", "1", "--n-decode", "0"));
        assertEquals(1, config.nPrefill);
        assertEquals(0, config.nDecode);
    }

    @Test
    void parseAllowsDecodeOnlyCluster() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--n-prefill", "0", "--n-decode", "3"));
        assertEquals(0, config.nPrefill);
        assertEquals(3, config.nDecode);
    }

    @Test
    void parseRejectsZeroEngineCluster() {
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--n-prefill", "0", "--n-decode", "0")));
    }

    @Test
    void parseRejectsNegativeCountsAndBadThreads() {
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--n-prefill", "-1", "--n-decode", "2")));
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--event-loop-threads", "0")));
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--completion-threads", "0")));
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--decode-max-concurrency", "0")));
    }

    // ──────────── New launcher parameters ────────────

    @Test
    void parseKeepsLegacyDefaultsWhenNewParamsAbsent() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(baseArgs());
        assertEquals(JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS, config.totalKvTokens);
        assertEquals(6_291_456L, config.totalKvTokens);
        // Capacity model v2 per-role pools: heterogeneous defaults (decode =
        // 2/3 of prefill) and 0 = derive blocks from ceil(total/spb).
        assertEquals(JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS, config.prefillTotalKvTokens,
                "prefill pool keeps the legacy default");
        assertEquals(JavaMockEngineCluster.DEFAULT_DECODE_TOTAL_KV_TOKENS,
                config.decodeTotalKvTokens,
                "decode pool defaults to the smaller heterogeneous capacity");
        assertEquals(4_194_304L, config.decodeTotalKvTokens);
        assertEquals(0, config.prefillCacheBlocks,
                "prefill-kv-pool-blocks defaults to 0 (derive from token capacity)");
        assertEquals(0, config.decodeCacheBlocks,
                "decode-kv-pool-blocks defaults to 0 (derive from token capacity)");
        assertEquals(0, config.blockSize, "block-size defaults to 0 (keep perf-file value)");
        assertEquals(JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY,
                config.decodeMaxConcurrency);
        assertEquals(128, config.decodeMaxConcurrency);
        assertEquals(5000, config.statsIntervalMs, "stats interval keeps the historical 5s cadence");
    }

    @Test
    void parseNewParamsOverrideDefaults() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(
                with(baseArgs(),
                        "--total-kv-tokens", "1234567",
                        "--block-size", "256",
                        "--decode-max-concurrency", "64",
                        "--stats-interval-ms", "1000"));
        assertEquals(1_234_567L, config.totalKvTokens);
        // Uniform knob applies to BOTH per-role pools (Python compat).
        assertEquals(1_234_567L, config.prefillTotalKvTokens);
        assertEquals(1_234_567L, config.decodeTotalKvTokens);
        assertEquals(256, config.blockSize);
        assertEquals(64, config.decodeMaxConcurrency);
        assertEquals(1000, config.statsIntervalMs);
    }

    @Test
    void parsePerRoleKvParamsOverrideIndependently() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(
                with(baseArgs(),
                        "--prefill-total-kv-tokens", "2000000",
                        "--decode-total-kv-tokens", "1000000",
                        "--prefill-kv-pool-blocks", "64",
                        "--decode-kv-pool-blocks", "32"));
        assertEquals(2_000_000L, config.prefillTotalKvTokens,
                "--prefill-total-kv-tokens overrides only the prefill pool");
        assertEquals(1_000_000L, config.decodeTotalKvTokens,
                "--decode-total-kv-tokens overrides only the decode pool");
        assertEquals(JavaMockEngineCluster.DEFAULT_TOTAL_KV_TOKENS, config.totalKvTokens,
                "the legacy uniform field stays untouched by per-role flags");
        assertEquals(64, config.prefillCacheBlocks,
                "--prefill-kv-pool-blocks overrides the prefill pool block count");
        assertEquals(32, config.decodeCacheBlocks,
                "--decode-kv-pool-blocks overrides the decode pool block count");
    }

    @Test
    void parseKvPoolBlocksNewNamesOverridePool() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(
                with(baseArgs(),
                        "--prefill-kv-pool-blocks", "64",
                        "--decode-kv-pool-blocks", "32"));
        assertEquals(64, config.prefillCacheBlocks,
                "--prefill-kv-pool-blocks sizes the prefill pool block count");
        assertEquals(32, config.decodeCacheBlocks,
                "--decode-kv-pool-blocks sizes the decode pool block count");
    }

    @Test
    void parseRejectsRetiredCacheBlocksNames() {
        // The pre-v2 names were removed outright: passing either one is an
        // unknown argument and must fail fast instead of silently aliasing.
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--prefill-cache-blocks", "99")));
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--decode-cache-blocks", "77")));
    }

    @Test
    void parseRejectsNonPositiveStatsInterval() {
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--stats-interval-ms", "0")));
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--stats-interval-ms", "-5")));
    }

    // ──────────── Single-role discovery files ────────────

    @Test
    void writeDiscoveryFilesHandlesPrefillOnlyCluster() throws Exception {
        Path endpointFile = tempDir.resolve("single/prefill-endpoint.json");
        Path envFile = tempDir.resolve("single/prefill-env.sh");
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(new String[]{
                "--endpoint-file", endpointFile.toString(),
                "--env-file", envFile.toString(),
                "--performance", tempDir.resolve("performance.json").toString(),
                "--master-config", tempDir.resolve("master.json").toString(),
                "--n-prefill", "2",
                "--n-decode", "0",
                "--base-grpc-port", "64000",
        });
        JavaMockEngineCluster.writeDiscoveryFiles(config);

        JsonNode payload = MAPPER.readTree(endpointFile.toFile());
        JsonNode engines = payload.get("engines");
        assertEquals(2, engines.size(), "only prefill engines should be listed");
        assertEquals("prefill-0", engines.get(0).get("name").asText());
        assertEquals("prefill-1", engines.get(1).get("name").asText());
        assertEquals(64000, engines.get(0).get("grpc_port").asInt());

        assertEquals(1, payload.get("env").size());
        JsonNode service = MAPPER.readTree(payload.get("env").get("MODEL_SERVICE_CONFIG").asText());
        assertEquals(config.discoveryFile, service.get("discovery_file").asText());
        assertTrue(Files.readString(envFile).contains("MODEL_SERVICE_CONFIG="));

        // Mirror check: decode-only cluster lists only decode engines.
        JavaMockEngineCluster.Config decodeOnly = JavaMockEngineCluster.Config.parse(new String[]{
                "--endpoint-file", tempDir.resolve("single/decode-endpoint.json").toString(),
                "--performance", tempDir.resolve("performance.json").toString(),
                "--master-config", tempDir.resolve("master.json").toString(),
                "--n-prefill", "0",
                "--n-decode", "1",
                "--base-grpc-port", "64000",
        });
        JavaMockEngineCluster.writeDiscoveryFiles(decodeOnly);
        JsonNode decodePayload = MAPPER.readTree(tempDir.resolve("single/decode-endpoint.json").toFile());
        assertEquals(1, decodePayload.get("engines").size());
        assertEquals("decode-0", decodePayload.get("engines").get(0).get("name").asText());
        assertEquals(64000, decodePayload.get("engines").get(0).get("grpc_port").asInt(),
                "decode ports start at base when there are no prefill engines");
    }
}
