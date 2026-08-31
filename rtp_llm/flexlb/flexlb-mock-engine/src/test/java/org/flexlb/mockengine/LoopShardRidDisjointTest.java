package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for the loop-mode cross-shard request_id collision bug:
 * with LOOP=1 every shard used to replay the FULL trace (no shard slicing) and
 * the loop suffix ("_L" + loopIdx) carried no shard identity, so all shards
 * emitted identical deterministic request_ids and the master rejected 7/8 of
 * them as "duplicate request_id".
 *
 * <p>Fix under test: (1) loop mode shard-slices the trace via shardSlice, and
 * (2) makeLoopRequest suffixes carry the shard index ("_S" + shard + "_L" + loop).
 */
class LoopShardRidDisjointTest {

    private static final int NUM_SHARDS = 8;

    @TempDir
    Path tempDir;

    private static JavaLoadClient.TraceRecord rec(int idx) {
        String sourceRid = "rid-" + idx;
        return new JavaLoadClient.TraceRecord(
                JavaLoadClient.stableRequestId(sourceRid), sourceRid, "trace-" + idx,
                idx * 100L, 128, 16, Collections.emptyList(), Collections.nCopies(128, 0));
    }

    private JavaLoadClient loopClient(int shardIndex) {
        JavaLoadClient.Config config = new JavaLoadClient.Config(
                "trace.jsonl", "127.0.0.1:7001", "127.0.0.1:7003",
                0, 16, 10.0, 1, tempDir.resolve("out").toString(), NUM_SHARDS, shardIndex, 0,
                120_000L, 500.0, "skip", false, true, 1, 1, 0L, 120, true,
                "engine_service", "",
                false, 10, 1000, 0, 0, "", false, "", true);
        return new JavaLoadClient(config);
    }

    // ------------------------------------------------------------------
    // shardSlice: pure i % numShards partition, no duration/limit filtering.
    // ------------------------------------------------------------------

    @Test
    void shardSlicePartitionsTraceWithoutOverlapOrLoss() {
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            records.add(rec(i));
        }

        Set<String> seen = new HashSet<>();
        int total = 0;
        for (int shard = 0; shard < NUM_SHARDS; shard++) {
            List<JavaLoadClient.TraceRecord> slice =
                    JavaLoadClient.shardSlice(records, NUM_SHARDS, shard);
            for (JavaLoadClient.TraceRecord r : slice) {
                assertTrue(seen.add(r.sourceRid), "record " + r.sourceRid
                        + " appears in more than one shard");
            }
            total += slice.size();
        }
        assertEquals(records.size(), total, "shards must cover the whole trace exactly once");
    }

    @Test
    void shardSliceSingleShardReturnsAllRecords() {
        List<JavaLoadClient.TraceRecord> records = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            records.add(rec(i));
        }
        assertEquals(10, JavaLoadClient.shardSlice(records, 1, 0).size());
    }

    // ------------------------------------------------------------------
    // End-to-end rid disjointness: 8 shards, same trace, loopIdx 0/1/2.
    // ------------------------------------------------------------------

    @Test
    void loopModeRidsAreDisjointAcrossShards() {
        List<JavaLoadClient.TraceRecord> trace = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            trace.add(rec(i));
        }

        List<Set<Long>> ridsPerShard = new ArrayList<>();
        for (int shard = 0; shard < NUM_SHARDS; shard++) {
            JavaLoadClient client = loopClient(shard);
            List<JavaLoadClient.TraceRecord> slice =
                    JavaLoadClient.shardSlice(trace, NUM_SHARDS, shard);
            Set<Long> rids = new HashSet<>();
            for (int loopIdx = 0; loopIdx <= 2; loopIdx++) {
                for (JavaLoadClient.TraceRecord r : slice) {
                    // Mirrors run(): loopIdx == 0 uses the original record.
                    JavaLoadClient.TraceRecord sent =
                            loopIdx == 0 ? r : client.makeLoopRequest(r, loopIdx, 0);
                    assertTrue(rids.add(sent.requestId),
                            "duplicate rid within shard " + shard + ": " + sent.sourceRid);
                }
            }
            ridsPerShard.add(rids);
        }

        for (int a = 0; a < NUM_SHARDS; a++) {
            for (int b = a + 1; b < NUM_SHARDS; b++) {
                Set<Long> overlap = new HashSet<>(ridsPerShard.get(a));
                overlap.retainAll(ridsPerShard.get(b));
                assertTrue(overlap.isEmpty(),
                        "shards " + a + " and " + b + " share " + overlap.size() + " rid(s)");
            }
        }
    }

    // ------------------------------------------------------------------
    // Defensive suffix: even on an UNSLICED trace, loopIdx >= 1 rids must
    // not collide across shards (guards against future slicing regressions).
    // ------------------------------------------------------------------

    @Test
    void loopSuffixCarriesShardIndexEvenWithoutSlicing() {
        List<JavaLoadClient.TraceRecord> trace = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            trace.add(rec(i));
        }

        List<Set<Long>> ridsPerShard = new ArrayList<>();
        for (int shard = 0; shard < NUM_SHARDS; shard++) {
            JavaLoadClient client = loopClient(shard);
            Set<Long> rids = new HashSet<>();
            for (int loopIdx = 1; loopIdx <= 2; loopIdx++) {
                for (JavaLoadClient.TraceRecord r : trace) {
                    JavaLoadClient.TraceRecord sent = client.makeLoopRequest(r, loopIdx, 0);
                    assertEquals(r.sourceRid + "_S" + shard + "_L" + loopIdx, sent.sourceRid);
                    rids.add(sent.requestId);
                }
            }
            ridsPerShard.add(rids);
        }

        for (int a = 0; a < NUM_SHARDS; a++) {
            for (int b = a + 1; b < NUM_SHARDS; b++) {
                Set<Long> overlap = new HashSet<>(ridsPerShard.get(a));
                overlap.retainAll(ridsPerShard.get(b));
                assertTrue(overlap.isEmpty(),
                        "unsliced shards " + a + " and " + b + " share rid(s): " + overlap);
            }
        }
    }
}
