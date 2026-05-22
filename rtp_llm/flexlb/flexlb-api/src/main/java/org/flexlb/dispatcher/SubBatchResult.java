package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.JsonNode;

/**
 * One sub-batch outcome: the FE's response JSON ({@code ok}) or a failure marker ({@code failed}).
 * Always carries the chunk size so a failed sub-batch can be padded with the right number of
 * placeholders during merge.
 */
public record SubBatchResult(JsonNode response, int chunkSize) {

    public static SubBatchResult ok(JsonNode response, int chunkSize) {
        return new SubBatchResult(response, chunkSize);
    }

    public static SubBatchResult failed(int chunkSize) {
        return new SubBatchResult(null, chunkSize);
    }

    public boolean isSuccess() {
        return response != null;
    }
}
