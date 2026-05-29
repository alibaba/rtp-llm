package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import java.util.ArrayList;
import java.util.List;

public final class BatchSplitter {

    private BatchSplitter() {
    }

    /**
     * Spec-aware entry point: dispatches to {@link #splitArray} for {@link SubBatchSpec.Mode#SIZE}
     * or {@link #splitByCount} for {@link SubBatchSpec.Mode#COUNT}. The single switch lives here so
     * callers (the real fanout handler and any inspection-style endpoints) don't each re-implement
     * the spec→method routing.
     */
    public static List<ArrayNode> split(ArrayNode arr, SubBatchSpec spec, ObjectMapper mapper) {
        return switch (spec.mode()) {
            case SIZE -> splitArray(arr, spec.value(), mapper);
            case COUNT -> splitByCount(arr, spec.value(), mapper);
        };
    }

    /**
     * Split a JSON array into ordered chunks of at most {@code chunkSize}. Order is preserved;
     * the final chunk may be smaller. An empty input yields no chunks. Items are shared by
     * reference with the input array; do not mutate them after splitting.
     */
    public static List<ArrayNode> splitArray(ArrayNode arr, int chunkSize, ObjectMapper mapper) {
        assert chunkSize >= 1 : "chunkSize must be >= 1, got " + chunkSize;
        int n = arr.size();
        if (n == 0) return List.of();
        int chunks = (n + chunkSize - 1) / chunkSize;
        List<ArrayNode> out = new ArrayList<>(chunks);
        for (int c = 0; c < chunks; c++) {
            ArrayNode chunk = mapper.createArrayNode();
            int start = c * chunkSize;
            int end = Math.min(start + chunkSize, n);
            for (int i = start; i < end; i++) chunk.add(arr.get(i));
            out.add(chunk);
        }
        return out;
    }

    /**
     * Split into at most {@code requestedCount} ordered chunks. Items distribute as evenly as
     * possible — the remainder lands on the leading chunks (so chunks 0..r-1 have one extra item).
     * If {@code total &lt; requestedCount} the count is clamped to {@code total} so no empty chunk
     * is emitted. Order is preserved; items are shared by reference with the input array, do not
     * mutate after splitting.
     */
    public static List<ArrayNode> splitByCount(ArrayNode arr, int requestedCount, ObjectMapper mapper) {
        assert requestedCount >= 1 : "requestedCount must be >= 1, got " + requestedCount;
        int n = arr.size();
        if (n == 0) return List.of();
        int chunks = Math.min(requestedCount, n);
        int base = n / chunks;
        int remainder = n % chunks;
        List<ArrayNode> out = new ArrayList<>(chunks);
        int cursor = 0;
        for (int c = 0; c < chunks; c++) {
            int size = base + (c < remainder ? 1 : 0);
            ArrayNode chunk = mapper.createArrayNode();
            for (int i = 0; i < size; i++) chunk.add(arr.get(cursor + i));
            out.add(chunk);
            cursor += size;
        }
        return out;
    }
}
