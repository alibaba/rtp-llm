package org.flexlb.dispatcher;

import com.alibaba.fastjson2.JSONArray;
import com.alibaba.fastjson2.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Generic merger for every batch endpoint on the dispatcher batch path. Picks the first
 * well-formed sub-batch's body as the response envelope template, replaces its
 * {@code spec.responseArrayField} with the stitched array, then optionally invokes
 * {@code spec.postMerger} for cross-chunk aggregation. Failed sub-batches are padded
 * item-by-item via {@code spec.failedItemFactory}, and an {@code _partial_failure} object
 * is appended when any items failed.
 */
public final class ResponseMerger {

    private ResponseMerger() {}

    /**
     * Merge outcome: the client-facing {@code body}, success/total counts, the absolute item
     * indices that failed, one reason string per failed chunk (in chunk order), and the HTTP
     * status to use when {@link #allFailed()} — the shared FE 4xx when every sub-batch failed
     * with the same client error, otherwise 500. The handler returns 200 on any success.
     */
    public record MergedResponse(JSONObject body,
                                 int succeededChunks,
                                 int totalChunks,
                                 List<Integer> failedIndices,
                                 List<String> failedReasons,
                                 int errorStatus) {
        public boolean allFailed() {
            return totalChunks > 0 && succeededChunks == 0;
        }
    }

    public static MergedResponse merge(List<SubBatchResult> subs, BatchEndpointSpec spec) {
        JSONObject envelope = null;
        int totalItems = 0;
        for (SubBatchResult s : subs) {
            totalItems += s.chunkSize();
            if (envelope == null && wellFormed(s, spec)) {
                // Top-level copy only: the template's fields (including the array slot we
                // overwrite next) land on a private map, while nested values stay shared with
                // the sub-body — which is dead after merge, so no isolation is lost and the
                // largest object on the merge path is never serialized+reparsed.
                envelope = new JSONObject(s.body());
                envelope.put(spec.getResponseArrayField(), new JSONArray());
            }
        }
        if (envelope == null) {
            List<String> reasons = new ArrayList<>(subs.size());
            for (SubBatchResult s : subs) {
                reasons.add(reasonFor(s));
            }
            return new MergedResponse(new JSONObject(), 0, subs.size(),
                    allIndices(totalItems), reasons, commonErrorStatus(subs));
        }
        JSONArray merged = envelope.getJSONArray(spec.getResponseArrayField());
        List<Integer> failedIndices = new ArrayList<>();
        List<String> failedReasons = new ArrayList<>();
        int succeededChunks = 0;
        for (SubBatchResult s : subs) {
            if (wellFormed(s, spec)) {
                JSONArray sourceArr = s.body().getJSONArray(spec.getResponseArrayField());
                merged.addAll(sourceArr);
                succeededChunks++;
            } else {
                String reason = reasonFor(s);
                failedReasons.add(reason);
                for (int i = 0; i < s.chunkSize(); i++) {
                    int abs = s.startIndex() + i;
                    merged.add(spec.getFailedItemFactory().build(abs, reason));
                    failedIndices.add(abs);
                }
            }
        }
        if (!failedIndices.isEmpty()) {
            JSONObject pf = new JSONObject();
            pf.put("failed_count", failedIndices.size());
            pf.put("total_count", totalItems);
            JSONArray fi = new JSONArray(failedIndices.size());
            fi.addAll(failedIndices);
            pf.put("failed_indices", fi);
            envelope.put("_partial_failure", pf);
        }
        if (spec.getPostMerger() != null) {
            spec.getPostMerger().apply(envelope, subs, failedIndices);
        }
        return new MergedResponse(envelope, succeededChunks, subs.size(), failedIndices, failedReasons, 500);
    }

    /**
     * HTTP status for the all-failed case: the shared FE status when every sub-batch failed with
     * the same 4xx (so a client error is not masked as a server 500), otherwise 500.
     */
    private static int commonErrorStatus(List<SubBatchResult> subs) {
        int common = -1;
        for (SubBatchResult s : subs) {
            int st = s.feStatus();
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

    private static String reasonFor(SubBatchResult s) {
        return s.success() ? "malformed_sub_batch" : s.reason();
    }

    private static boolean wellFormed(SubBatchResult s, BatchEndpointSpec spec) {
        if (!s.success() || s.body() == null) {
            return false;
        }
        JSONArray arr = s.body().getJSONArray(spec.getResponseArrayField());
        return arr != null && arr.size() == s.chunkSize();
    }

    private static List<Integer> allIndices(int n) {
        List<Integer> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(i);
        }
        return out;
    }
}
