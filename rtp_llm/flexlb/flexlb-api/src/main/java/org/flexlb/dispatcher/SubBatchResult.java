package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;

/**
 * Outcome of a single sub-batch fanout on the dispatcher batch path. Mirrors
 * {@code FanoutService.SubBatchResult} on the Jackson side but carries a fastjson2
 * {@link JSONObject} body, so the merger never has to bridge JSON libraries.
 *
 * <p>{@code body} is the FE response parsed via fastjson2 on success; {@code null} on failure.
 * {@code startIndex} is the absolute offset of this chunk's first item in the original request
 * array — preserved here so per-item failure placeholders can land at the right absolute index
 * after merge.
 */
public record SubBatchResult(boolean success,
                                      JSONObject body,
                                      int chunkSize,
                                      int startIndex,
                                      String reason) {

    public static SubBatchResult ok(JSONObject body, int chunkSize, int startIndex) {
        return new SubBatchResult(true, body, chunkSize, startIndex, null);
    }

    public static SubBatchResult failed(int chunkSize, int startIndex, String reason) {
        return new SubBatchResult(false, null, chunkSize, startIndex, reason);
    }
}
