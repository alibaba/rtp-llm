package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

/**
 * Result of merging sub-batches: the client-facing {@code body}, success/total counts, and the
 * absolute item indices that failed. The handler returns 200 on any success and reserves 500 for
 * the all-failed case.
 */
public record MergedResponse(ObjectNode body,
                             int succeededChunks,
                             int totalChunks,
                             List<Integer> failedIndices) {

    /** @deprecated three-arg constructor for legacy callers; removed in V7. Use the canonical constructor with {@code failedIndices}. */
    @Deprecated
    public MergedResponse(ObjectNode body, int succeededChunks, int totalChunks) {
        this(body, succeededChunks, totalChunks, List.of());
    }

    public boolean allFailed() {
        return totalChunks > 0 && succeededChunks == 0;
    }
}
