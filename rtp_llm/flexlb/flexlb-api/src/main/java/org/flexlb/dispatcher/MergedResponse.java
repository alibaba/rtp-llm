package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.node.ObjectNode;

/**
 * Result of merging sub-batches: the client-facing {@code body} plus how many sub-batches actually
 * succeeded. The handler returns 200 on any success and reserves 500 for the all-failed case.
 */
public record MergedResponse(ObjectNode body, int succeededChunks, int totalChunks) {

    public boolean allFailed() {
        return totalChunks > 0 && succeededChunks == 0;
    }
}
