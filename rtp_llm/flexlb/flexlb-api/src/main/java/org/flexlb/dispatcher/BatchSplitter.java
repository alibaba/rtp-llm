package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;

import java.util.ArrayList;
import java.util.List;

public final class BatchSplitter {

    private BatchSplitter() {
    }

    /**
     * Split prompts into ordered chunks of at most {@code subBatchSize}. Order is preserved;
     * the final chunk may be smaller.
     */
    public static List<List<String>> split(List<String> prompts, int subBatchSize) {
        assert subBatchSize >= 1 : "subBatchSize must be >= 1, got " + subBatchSize;
        List<List<String>> chunks = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i += subBatchSize) {
            chunks.add(new ArrayList<>(prompts.subList(i, Math.min(i + subBatchSize, prompts.size()))));
        }
        return chunks;
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
