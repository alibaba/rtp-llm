package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;

/**
 * Outcome of a single sub-batch fanout on the dispatcher batch path.
 *
 * <p>{@code body} is the FE response parsed via fastjson2 on success; {@code null} on failure.
 * {@code startIndex} is the absolute offset of this chunk's first item in the original request
 * array — preserved here so per-item failure placeholders can land at the right absolute index
 * after merge. {@code feStatus} is the FE response status when the sub-call got an HTTP error
 * response, or 0 when the failure was not an HTTP status (connect error, FE pick failure); it
 * lets the all-failed path surface a shared FE 4xx instead of masking it as a 500.
 */
public record SubBatchResult(boolean success,
                             JSONObject body,
                             int chunkSize,
                             int startIndex,
                             String reason,
                             int feStatus) {

    public static SubBatchResult ok(JSONObject body, int chunkSize, int startIndex) {
        return new SubBatchResult(true, body, chunkSize, startIndex, null, 200);
    }

    public static SubBatchResult failed(int chunkSize, int startIndex, String reason) {
        return failed(chunkSize, startIndex, reason, 0);
    }

    public static SubBatchResult failed(int chunkSize, int startIndex, String reason, int feStatus) {
        return new SubBatchResult(false, null, chunkSize, startIndex, reason, feStatus);
    }
}
