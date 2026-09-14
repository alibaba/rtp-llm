package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;

import java.util.ArrayList;
import java.util.List;

/** Merges the ordered results emitted by FanoutService, padding failures at their input indices. */
public final class ResponseMerger {
    private ResponseMerger() {}

    public record MergedResponse(int status, JSONObject body) {
    }

    public static MergedResponse merge(List<SubBatchResult> subs, BatchEndpointSpec spec,
                                       JSONObject originalRequest) {
        JSONObject envelope = null;
        JSONArray merged = new JSONArray();
        List<Integer> failedIndices = new ArrayList<>();
        List<String> failedReasons = new ArrayList<>();
        int totalItems = 0;
        for (SubBatchResult sub : subs) {
            totalItems += sub.chunkSize();
            if (wellFormed(sub, spec)) {
                if (envelope == null) {
                    envelope = new JSONObject(sub.body());
                }
                merged.addAll(sub.body().getJSONArray(spec.getResponseArrayField()));
            } else {
                String reason = reasonFor(sub);
                failedReasons.add(reason);
                for (int i = 0; i < sub.chunkSize(); i++) {
                    int index = sub.startIndex() + i;
                    merged.add(spec.failedItem(index, reason));
                    failedIndices.add(index);
                }
            }
        }
        boolean allFailed = envelope == null && !subs.isEmpty();
        if (allFailed || (spec.isFailOnPartialFailure() && !failedIndices.isEmpty())) {
            return new MergedResponse(allFailed ? commonErrorStatus(subs) : 500,
                    JSONObject.of("error", allFailed ? "all_sub_batches_failed" : "sub_batch_failed",
                            "failed_count", failedIndices.size(), "total_count", totalItems,
                            "total_chunks", subs.size(),
                            "failed_reasons", failedReasons.stream().distinct().toList()));
        }
        if (envelope == null) {
            envelope = new JSONObject();
        }
        envelope.put(spec.getResponseArrayField(), merged);
        if (!failedIndices.isEmpty()) {
            envelope.put("_partial_failure", JSONObject.of("failed_count", failedIndices.size(),
                    "total_count", totalItems, "failed_indices", new JSONArray(failedIndices)));
        }
        spec.finishMerge(envelope, subs, originalRequest);
        return new MergedResponse(200, envelope);
    }

    /** On total failure, use the shared FE 4xx if all contacted FEs return it; otherwise 500. */
    private static int commonErrorStatus(List<SubBatchResult> subs) {
        int common = -1;
        for (SubBatchResult s : subs) {
            int st = s.feStatus();
            if (st <= 0) {
                continue;
            }
            if (st < 400 || st > 499) {
                return 500;
            }
            if (common == -1) {
                common = st;
            } else if (common != st) {
                return 500;
            }
        }
        return common == -1 ? 500 : common;
    }

    /** Client-facing failure reason: a stable, bounded code — never the raw exception text. */
    private static String reasonFor(SubBatchResult s) {
        if (s.success()) {
            return "malformed_sub_batch";
        }
        int st = s.feStatus();
        if (st >= 400 && st < 500) {
            return "fe_client_error";
        }
        if (st >= 500) {
            return "fe_server_error";
        }
        return "fe_unavailable";
    }

    static boolean wellFormed(SubBatchResult s, BatchEndpointSpec spec) {
        if (!s.success() || s.body() == null) {
            return false;
        }
        JSONArray arr = s.body().getJSONArray(spec.getResponseArrayField());
        return arr != null && arr.size() == s.chunkSize();
    }
}
