package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * One sub-batch outcome. Successful chunks carry the FE's response JSON in {@code body};
 * failed chunks carry a textual {@code reason}. {@code startIndex} is the absolute offset of
 * this chunk's first item in the full batch and is used by {@link PartialFailureMerger} when
 * building per-item failure placeholders / failed_indices metadata.
 */
public record SubBatchResult(boolean success,
                             int chunkSize,
                             int startIndex,
                             JsonNode body,
                             String reason) {

    public static SubBatchResult ok(JsonNode body, int chunkSize, int startIndex) {
        return new SubBatchResult(true, chunkSize, startIndex, body, null);
    }

    public static SubBatchResult failed(int chunkSize, int startIndex, String reason) {
        return new SubBatchResult(false, chunkSize, startIndex, null, reason);
    }

    public boolean isSuccess() {
        return success;
    }
}
