package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONObject;

/** One chunk's body, input position and FE status (zero if no FE response arrived). */
public record SubBatchResult(JSONObject body, int chunkSize, int startIndex, int feStatus) {
    public boolean success() {
        return feStatus >= 200 && feStatus < 300;
    }

    public static SubBatchResult ok(JSONObject body, int chunkSize, int startIndex) {
        return new SubBatchResult(body, chunkSize, startIndex, 200);
    }

    public static SubBatchResult failed(int chunkSize, int startIndex, int feStatus) {
        return new SubBatchResult(null, chunkSize, startIndex, feStatus);
    }
}
