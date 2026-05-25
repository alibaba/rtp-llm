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

    /** @deprecated use {@link #ok(JsonNode, int, int)} with an absolute startIndex; removed in V7. */
    @Deprecated
    public static SubBatchResult ok(JsonNode body, int chunkSize) {
        return new SubBatchResult(true, chunkSize, -1, body, null);
    }

    /** @deprecated use {@link #failed(int, int, String)} with absolute startIndex and reason; removed in V7. */
    @Deprecated
    public static SubBatchResult failed(int chunkSize) {
        return new SubBatchResult(false, chunkSize, -1, null, "unknown");
    }

    public boolean isSuccess() {
        return success;
    }

    /** @deprecated legacy alias for {@link #body()}; removed in V7 once {@link ResponseMerger} is deleted. */
    @Deprecated
    public JsonNode response() {
        return body;
    }
}
