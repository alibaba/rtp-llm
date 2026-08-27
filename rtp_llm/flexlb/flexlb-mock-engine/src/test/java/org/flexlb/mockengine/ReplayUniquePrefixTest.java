package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

/**
 * Regression tests for the loop-mode cache-affinity routing collapse:
 * makeLoopRequest used to pass blockKeys through unchanged, so every replay
 * round presented byte-identical block_cache_keys and the master's cache
 * affinity routed each rid to the SAME prefill engine round after round
 * (P-side Gini ~0.56 across 750 engines).
 *
 * <p>Fix under test: REPLAY_UNIQUE_PREFIX (default on) re-salts only
 * blockKeys[0] with roundSaltedKey(k0, loopIdx), keeping the shared suffix
 * blocks (cross-request prefix reuse) while giving every loop round a
 * unique routing prefix.
 */
class ReplayUniquePrefixTest {

    @TempDir
    Path tempDir;

    private static JavaLoadClient.TraceRecord rec(List<Long> blockKeys) {
        return new JavaLoadClient.TraceRecord(
                1L, "rid-0", "trace-0", 0L, 128, 16, blockKeys, null);
    }

    private JavaLoadClient client(boolean replayUniquePrefix) {
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:7003",
                0, 16, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
                120_000L, 500.0, "skip", false, true, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true,
                0, 0, "replay", 0.0, replayUniquePrefix);
        return new JavaLoadClient(config);
    }

    @Test
    void loopRoundSaltsFirstBlockKeyOnly() {
        List<Long> keys = List.of(111L, 222L, 333L);
        JavaLoadClient.TraceRecord req = rec(keys);
        JavaLoadClient.TraceRecord sent = client(true).makeLoopRequest(req, 1, 0);

        assertNotEquals(keys.get(0), sent.blockKeys.get(0),
                "loop round must re-salt blockKeys[0]");
        assertEquals(keys.get(1), sent.blockKeys.get(1),
                "suffix keys must be preserved for cross-request reuse");
        assertEquals(keys.get(2), sent.blockKeys.get(2),
                "suffix keys must be preserved for cross-request reuse");
        assertEquals(keys.size(), sent.blockKeys.size(),
                "key list length must not change");
        assertEquals(keys, req.blockKeys,
                "the source record's shared key list must never be mutated in place");
    }

    @Test
    void saltedKeyIsDeterministicPerRound() {
        JavaLoadClient.TraceRecord req = rec(List.of(42L, 43L));
        JavaLoadClient client = client(true);

        JavaLoadClient.TraceRecord first = client.makeLoopRequest(req, 1, 0);
        JavaLoadClient.TraceRecord again = client.makeLoopRequest(req, 1, 0);
        assertEquals(first.blockKeys, again.blockKeys,
                "same loop round must deterministically produce the same salted key");
        assertEquals(JavaLoadClient.roundSaltedKey(42L, 1), first.blockKeys.get(0),
                "salted key must equal roundSaltedKey(k0, loopIdx)");

        JavaLoadClient.TraceRecord nextRound = client.makeLoopRequest(req, 2, 0);
        assertNotEquals(first.blockKeys.get(0), nextRound.blockKeys.get(0),
                "different loop rounds must present different routing prefixes");
    }

    @Test
    void disabledSwitchPassesBlockKeysThrough() {
        List<Long> keys = List.of(111L, 222L, 333L);
        JavaLoadClient.TraceRecord req = rec(keys);
        JavaLoadClient.TraceRecord sent = client(false).makeLoopRequest(req, 1, 0);

        assertEquals(keys, sent.blockKeys,
                "REPLAY_UNIQUE_PREFIX=false must keep the original block keys");
    }

    @Test
    void emptyBlockKeysStayEmptyWhenEnabled() {
        JavaLoadClient.TraceRecord sent = client(true)
                .makeLoopRequest(rec(List.of()), 1, 0);
        assertEquals(0, sent.blockKeys.size(),
                "empty key lists must pass through untouched");
    }

    @Test
    void defaultConstructorKeepsUniquePrefixEnabled() {
        // The 32-arg convenience constructor (used by existing tests) defaults
        // to enabled, mirroring the env default REPLAY_UNIQUE_PREFIX=true.
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:7003",
                0, 16, 10.0, 1, tempDir.resolve("out").toString(), 1, 0, 0,
                120_000L, 500.0, "skip", false, true, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true);
        JavaLoadClient.TraceRecord sent = new JavaLoadClient(config)
                .makeLoopRequest(rec(List.of(7L)), 1, 0);
        assertNotEquals(7L, sent.blockKeys.get(0),
                "convenience constructor must default REPLAY_UNIQUE_PREFIX to on");
    }
}
