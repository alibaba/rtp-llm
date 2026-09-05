package org.flexlb.discovery;

import org.flexlb.dao.master.WorkerHost;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for {@link FileServiceDiscovery} read-side policy:
 * normal parse / missing domain / file-missing fallback / corrupted-JSON
 * fallback / hot update takes effect / first-read failure throws /
 * partial invalid entry rejects the whole snapshot.
 */
class FileServiceDiscoveryTest {

    private static final String PREFILL_DOMAIN = "mock.prefill.hosts.address";
    private static final String DECODE_DOMAIN = "mock.decode.hosts.address";

    @TempDir
    Path tempDir;

    @Test
    void parsesHostsForKnownDomains() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{"
                + "\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\", \"127.0.0.1:55151\"],"
                + "\"" + DECODE_DOMAIN + "\": [\"127.0.0.1:55160\"]"
                + "}");

        FileServiceDiscovery discovery = new FileServiceDiscovery(file);

        List<WorkerHost> prefill = discovery.getHosts(PREFILL_DOMAIN);
        assertEquals(2, prefill.size());
        assertEquals("127.0.0.1:55150", prefill.get(0).getIpPort());
        assertEquals("127.0.0.1:55151", prefill.get(1).getIpPort());

        List<WorkerHost> decode = discovery.getHosts(DECODE_DOMAIN);
        assertEquals(1, decode.size());
        assertEquals("127.0.0.1:55160", decode.get(0).getIpPort());

        // HTTP port is the stored port; the gRPC derivation stays with the caller.
        assertEquals(55160, decode.get(0).getPort());
    }

    @Test
    void unknownDomainReturnsEmptyList() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\"]}");

        FileServiceDiscovery discovery = new FileServiceDiscovery(file);

        assertTrue(discovery.getHosts("mock.unknown.hosts.address").isEmpty(),
                "a valid file without the requested domain must return an empty list (NoOp parity)");
    }

    @Test
    void missingFileFallsBackToLastGoodSnapshot() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\"]}");
        FileServiceDiscovery discovery = new FileServiceDiscovery(file);
        assertEquals(1, discovery.getHosts(PREFILL_DOMAIN).size());

        Files.delete(file);

        List<WorkerHost> hosts = discovery.getHosts(PREFILL_DOMAIN);
        assertEquals(1, hosts.size(), "missing file must serve the last good snapshot, not an empty list");
        assertEquals("127.0.0.1:55150", hosts.get(0).getIpPort());
    }

    @Test
    void corruptedJsonFallsBackToLastGoodSnapshot() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\"]}");
        FileServiceDiscovery discovery = new FileServiceDiscovery(file);
        assertEquals(1, discovery.getHosts(PREFILL_DOMAIN).size());

        // A partial write window caught mid-rename: truncated JSON.
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\", \"127.0.0.");

        List<WorkerHost> hosts = discovery.getHosts(PREFILL_DOMAIN);
        assertEquals(1, hosts.size(), "corrupted JSON must fall back to the last good snapshot");
        assertEquals("127.0.0.1:55150", hosts.get(0).getIpPort());
    }

    @Test
    void partiallyInvalidEntryRejectsWholeSnapshot() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\"]}");
        FileServiceDiscovery discovery = new FileServiceDiscovery(file);
        assertEquals(1, discovery.getHosts(PREFILL_DOMAIN).size());

        // New file has one valid + one invalid entry — the whole parse must fail
        // so callers never observe a half list.
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\", \"not-a-host\"]}");

        List<WorkerHost> hosts = discovery.getHosts(PREFILL_DOMAIN);
        assertEquals(1, hosts.size(), "a snapshot with any invalid entry must be rejected entirely");
        assertEquals("127.0.0.1:55150", hosts.get(0).getIpPort());
    }

    @Test
    void hotUpdateTakesEffectOnNextCall() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\"]}");
        FileServiceDiscovery discovery = new FileServiceDiscovery(file);
        assertEquals(1, discovery.getHosts(PREFILL_DOMAIN).size());

        // Atomic-rename style hot update: write a tmp sibling then move over.
        writeAtomically(file, "{"
                + "\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\", \"127.0.0.1:55151\", \"127.0.0.1:55152\"],"
                + "\"" + DECODE_DOMAIN + "\": [\"127.0.0.1:55160\"]"
                + "}");

        List<WorkerHost> hosts = discovery.getHosts(PREFILL_DOMAIN);
        assertEquals(3, hosts.size(), "hot update must be visible on the next getHosts call");
        assertEquals("127.0.0.1:55152", hosts.get(2).getIpPort());
        assertEquals(1, discovery.getHosts(DECODE_DOMAIN).size());
    }

    @Test
    void recoveryAfterTransientCorruptionServesNewSnapshot() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\"]}");
        FileServiceDiscovery discovery = new FileServiceDiscovery(file);
        assertEquals(1, discovery.getHosts(PREFILL_DOMAIN).size());

        writeAtomically(file, "{corrupt");
        assertEquals(1, discovery.getHosts(PREFILL_DOMAIN).size(), "fallback during corruption window");

        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55200\"]}");
        assertEquals("127.0.0.1:55200", discovery.getHosts(PREFILL_DOMAIN).get(0).getIpPort(),
                "once the file is valid again the new snapshot must win");
    }

    @Test
    void firstReadFailureThrowsInsteadOfEmptyList() {
        Path file = tempDir.resolve("does-not-exist.json");
        FileServiceDiscovery discovery = new FileServiceDiscovery(file);

        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> discovery.getHosts(PREFILL_DOMAIN),
                "first read failure with no fallback must throw (misconfiguration should be loud)");
        assertNotNull(e.getMessage());
        assertTrue(e.getMessage().contains("no previous snapshot"),
                "exception should explain the missing-fallback condition");
    }

    @Test
    void blankAddressReturnsEmptyList() throws Exception {
        Path file = tempDir.resolve("discovery.json");
        writeAtomically(file, "{\"" + PREFILL_DOMAIN + "\": [\"127.0.0.1:55150\"]}");
        FileServiceDiscovery discovery = new FileServiceDiscovery(file);

        assertTrue(discovery.getHosts("  ").isEmpty());
        assertTrue(discovery.getHosts(null).isEmpty());
    }

    /** Write via tmp + atomic move, mirroring the writer-side protocol. */
    private static void writeAtomically(Path file, String content) throws Exception {
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.writeString(tmp, content, StandardCharsets.UTF_8);
        try {
            Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
        } catch (java.nio.file.AtomicMoveNotSupportedException e) {
            Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING);
        }
    }
}
