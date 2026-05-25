package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import java.util.ArrayList;
import java.util.List;

public final class BatchSplitter {

    private BatchSplitter() {
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
}
