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
        if (subBatchSize < 1) {
            throw new IllegalArgumentException("subBatchSize must be >= 1, got " + subBatchSize);
        }
        List<List<String>> chunks = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i += subBatchSize) {
            chunks.add(new ArrayList<>(prompts.subList(i, Math.min(i + subBatchSize, prompts.size()))));
        }
        return chunks;
    }

    /**
     * Number of chunks {@link #split} produces for {@code promptCount} prompts — the single source
     * of truth for the chunk count (e.g. sizing pre-assign's {@code /batch_schedule} call), so the
     * handler never recomputes {@code ceil(N/K)} independently of the split.
     */
    public static int chunkCount(int promptCount, int subBatchSize) {
        if (subBatchSize < 1) {
            throw new IllegalArgumentException("subBatchSize must be >= 1, got " + subBatchSize);
        }
        return (promptCount + subBatchSize - 1) / subBatchSize;
    }
}
