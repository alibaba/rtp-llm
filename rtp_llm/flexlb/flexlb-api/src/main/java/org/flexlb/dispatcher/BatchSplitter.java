package org.flexlb.dispatcher;

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
}
