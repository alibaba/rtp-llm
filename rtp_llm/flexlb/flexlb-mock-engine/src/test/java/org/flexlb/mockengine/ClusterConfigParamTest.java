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
        assertEquals(0, config.blockSize, "block-size defaults to 0 (keep perf-file value)");
        assertEquals(JavaMockEngineCluster.DEFAULT_DECODE_MAX_CONCURRENCY,
                config.decodeMaxConcurrency);
        assertEquals(132, config.decodeMaxConcurrency);
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
        assertEquals(256, config.blockSize);
        assertEquals(64, config.decodeMaxConcurrency);
        assertEquals(1000, config.statsIntervalMs);
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

        String envContent = Files.readString(envFile);
        assertTrue(envContent.contains("DOMAIN_ADDRESS:" + config.prefillDomain),
                "env file should define the prefill domain");
        // Decode domain record exists but with an empty address list.
        assertTrue(envContent.contains("DOMAIN_ADDRESS:" + config.decodeDomain + "='"),
                "decode domain should be present with empty addresses");

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
