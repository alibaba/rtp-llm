package org.flexlb.constraint;

import org.flexlb.constraint.ConstraintTreeModels.Artifact;
import org.flexlb.constraint.ConstraintTreeModels.BuildRequest;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Opt-in scale validation for the production token-array input path. Enable it
 * with CONSTRAINT_TREE_RUN_SCALE_TEST=1 so ordinary unit-test runs stay fast.
 */
class ConstraintTreeScaleTest {

    @Test
    void buildsAndEncodesOneMillionVariableLengthSids() {
        assumeTrue("1".equals(System.getenv("CONSTRAINT_TREE_RUN_SCALE_TEST")),
                "set CONSTRAINT_TREE_RUN_SCALE_TEST=1 to run the one-million-SID test");

        Runtime runtime = Runtime.getRuntime();
        long heapBeforeInput = usedHeap(runtime);
        List<int[]> tokenIds = millionVariableLengthSids();
        long heapAfterInput = usedHeap(runtime);
        long buildStarted = System.nanoTime();

        Artifact artifact;
        byte[] payload;
        try (ConstraintTreeBuilder builder = new ConstraintTreeBuilder()) {
            artifact = builder.build(new BuildRequest(
                    2026090301L,
                    "gul_item",
                    1699,
                    151645,
                    "_",
                    tokenIds,
                    null));
            long buildFinished = System.nanoTime();
            payload = ConstraintTreeCsrCodec.encode(artifact);
            long encodeFinished = System.nanoTime();

            System.out.printf(
                    "constraint-tree-scale: build_ms=%.3f encode_ms=%.3f input_heap_mib=%.2f total_heap_delta_mib=%.2f payload_mib=%.2f%n",
                    nanosToMillis(buildFinished - buildStarted),
                    nanosToMillis(encodeFinished - buildFinished),
                    bytesToMiB(heapAfterInput - heapBeforeInput),
                    bytesToMiB(usedHeap(runtime) - heapBeforeInput),
                    bytesToMiB(payload.length));
        }

        assertEquals(1_000_000L, artifact.sidCount());
        assertEquals(1_080_854L, artifact.prefixCount());
        assertEquals(2_080_853L, artifact.edgeCount());
        assertEquals(20_970_292, payload.length);
    }

    private static List<int[]> millionVariableLengthSids() {
        final int count = 1_000_000;
        final int firstTokenFanout = 30_853;
        int[][] sids = new int[count][];

        for (int i = 0; i < count; i++) {
            int first = 152_000 + i % firstTokenFanout;
            int second = 185_000 + i / firstTokenFanout;
            int mod = i % 100;
            if (mod == 0) {
                sids[i] = new int[]{first};
            } else if (mod < 95) {
                sids[i] = new int[]{first, second};
            } else if (mod < 99) {
                sids[i] = new int[]{first, second, 210_000 + i % 997};
            } else {
                sids[i] = new int[]{first, second, 210_000 + i % 997, 220_000 + i % 991};
            }
        }
        return Arrays.asList(sids);
    }

    private static long usedHeap(Runtime runtime) {
        return runtime.totalMemory() - runtime.freeMemory();
    }

    private static double nanosToMillis(long nanos) {
        return nanos / 1_000_000.0;
    }

    private static double bytesToMiB(long bytes) {
        return bytes / (1024.0 * 1024.0);
    }
}
