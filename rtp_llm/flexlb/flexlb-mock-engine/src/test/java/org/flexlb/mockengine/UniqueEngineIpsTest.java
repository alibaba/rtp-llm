package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unique per-engine loopback advertisement IPs ({@code --unique-engine-ips},
 * default on): every engine used to declare host 127.0.0.1, so the master-side
 * engineIp Prometheus label had a single variant and per-engine gauge series
 * overwrote each other. These tests pin the derivation formula, the CLI
 * switch (both space-separated and glued {@code =false} forms), and the
 * discovery-file wiring (DOMAIN_ADDRESS / endpoints.json keep the HTTP port
 * convention grpcPort-1 with a unique host per engine).
 */
class UniqueEngineIpsTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    @TempDir
    Path tempDir;

    // ──────────── Derivation formula ────────────

    @Test
    void derivationIsUniqueAndWellFormedFor1250Engines() {
        Set<String> seen = new HashSet<>(1250);
        for (int idx = 0; idx < 1250; idx++) {
            String ip = JavaMockEngineCluster.derivedLoopbackIp(idx);
            String[] octets = ip.split("\\.");
            assertEquals(4, octets.length, "loopback advertisement is a /8-style 127.a.b.1: " + ip);
            assertEquals("127", octets[0]);
            assertEquals("1", octets[3], "the fixed 4th octet keeps the IP a full valid IPv4: " + ip);
            int third = Integer.parseInt(octets[1]);
            int fourth = Integer.parseInt(octets[2]);
            assertTrue(third >= 1 && third <= 5,
                    "third octet must stay out of the real 127.0.0.x range: " + ip);
            assertTrue(fourth >= 0 && fourth <= 249, "fourth octet must stay below 250: " + ip);
            assertTrue(seen.add(ip), "duplicate advertisement IP at engine index " + idx + ": " + ip);
        }
        assertEquals(1250, seen.size(), "750P + 500D production shape needs 1250 distinct IPs");
    }

    @Test
    void derivationCoversKnownBoundaries() {
        assertEquals("127.1.0.1", JavaMockEngineCluster.derivedLoopbackIp(0));
        assertEquals("127.1.249.1", JavaMockEngineCluster.derivedLoopbackIp(249));
        assertEquals("127.2.0.1", JavaMockEngineCluster.derivedLoopbackIp(250));
        // 750P + 500D: the first decode engine continues the global index.
        assertEquals("127.4.0.1", JavaMockEngineCluster.derivedLoopbackIp(750));
        assertEquals("127.5.249.1", JavaMockEngineCluster.derivedLoopbackIp(1249));
    }

    @Test
    void derivationRejectsOutOfRangeIndexes() {
        assertThrows(IllegalArgumentException.class,
                () -> JavaMockEngineCluster.derivedLoopbackIp(-1));
        assertThrows(IllegalArgumentException.class,
                () -> JavaMockEngineCluster.derivedLoopbackIp(63_750));
    }

    // ──────────── CLI switch ────────────

    @Test
    void uniqueEngineIpsDefaultsToTrue() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(baseArgs());
        assertTrue(config.uniqueEngineIps, "unique engine advertisement IPs must be on by default");
    }

    @Test
    void spaceFormDisablesUniqueEngineIps() {
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--unique-engine-ips", "false"));
        assertFalse(config.uniqueEngineIps);
    }

    @Test
    void gluedFormDisablesUniqueEngineIpsAndDoesNotSwallowNextArgument() {
        JavaMockEngineCluster.Config off = JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--unique-engine-ips=false"));
        assertFalse(off.uniqueEngineIps);
        // Glued true must not consume the following flag's value.
        JavaMockEngineCluster.Config on = JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--unique-engine-ips=true", "--stats-interval-ms", "2000"));
        assertTrue(on.uniqueEngineIps);
        assertEquals(2000, on.statsIntervalMs);
    }

    @Test
    void booleanFlagRejectsGarbageValues() {
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--unique-engine-ips", "yes")));
        assertThrows(IllegalArgumentException.class, () -> JavaMockEngineCluster.Config.parse(
                with(baseArgs(), "--unique-engine-ips=maybe")));
    }

    // ──────────── Discovery files carry the declared IPs ────────────

    @Test
    void writeDiscoveryFilesUsesUniqueIpsByDefault() throws Exception {
        JsonNode payload = writeDiscovery("--n-prefill", "2", "--n-decode", "2");

        JsonNode engines = payload.get("engines");
        assertEquals(4, engines.size());
        assertEquals("127.1.0.1", engines.get(0).get("ip").asText());
        assertEquals("127.1.0.1:64000", engines.get(0).get("grpc_addr").asText());
        assertEquals("127.1.0.1:63999", engines.get(0).get("http_addr").asText());
        assertEquals("127.1.1.1:64001", engines.get(1).get("grpc_addr").asText());
        assertEquals("127.1.2.1", engines.get(2).get("ip").asText(),
                "decode engines continue the global index after prefill");
        assertEquals("127.1.2.1:64002", engines.get(2).get("grpc_addr").asText());
        assertEquals("127.1.3.1:64003", engines.get(3).get("grpc_addr").asText());

        // DOMAIN_ADDRESS keeps the HTTP-port convention (grpcPort - 1); the
        // master-side http-protocol conversion and JavaLoadClient add +1 back.
        String prefillEnv = payload.get("env")
                .get("DOMAIN_ADDRESS:mock.prefill.hosts.address").asText();
        assertEquals("127.1.0.1:63999,127.1.1.1:64000", prefillEnv);
        String decodeEnv = payload.get("env")
                .get("DOMAIN_ADDRESS:mock.decode.hosts.address").asText();
        assertEquals("127.1.2.1:64001,127.1.3.1:64002", decodeEnv);
    }

    @Test
    void writeDiscoveryFilesFallsBackToConfigHostWhenDisabled() throws Exception {
        JsonNode payload = writeDiscovery("--n-prefill", "2", "--n-decode", "1",
                "--unique-engine-ips", "false");

        JsonNode engines = payload.get("engines");
        assertEquals(3, engines.size());
        assertEquals("127.0.0.1", engines.get(0).get("ip").asText());
        assertEquals("127.0.0.1:64000", engines.get(0).get("grpc_addr").asText());
        assertEquals("127.0.0.1:63999", engines.get(0).get("http_addr").asText());
        String prefillEnv = payload.get("env")
                .get("DOMAIN_ADDRESS:mock.prefill.hosts.address").asText();
        assertEquals("127.0.0.1:63999,127.0.0.1:64000", prefillEnv);
    }

    private JsonNode writeDiscovery(String... extra) throws Exception {
        Path endpointFile = tempDir.resolve("endpoint.json");
        JavaMockEngineCluster.Config config = JavaMockEngineCluster.Config.parse(with(new String[]{
                "--endpoint-file", endpointFile.toString(),
                "--performance", tempDir.resolve("performance.json").toString(),
                "--master-config", tempDir.resolve("master.json").toString(),
                "--base-grpc-port", "64000",
        }, extra));
        JavaMockEngineCluster.writeDiscoveryFiles(config);
        return MAPPER.readTree(endpointFile.toFile());
    }

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
}
