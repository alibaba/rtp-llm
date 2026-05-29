package org.flexlb.dispatcher;

import org.flexlb.dispatcher.FanoutService.SubBatchResult;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.ArrayList;
import java.util.List;

/**
 * Generic merger for every batch endpoint. Uses the first successful sub-batch's body as the
 * response envelope template, replaces its {@code spec.responseArrayField} with the stitched
 * array, then optionally invokes {@code spec.postMerger} for cross-chunk aggregation
 * (e.g. embedding {@code usage} sum). Failed sub-batches are padded item-by-item via
 * {@code spec.failedItemFactory}, and an {@code _partial_failure} object is appended when any
 * items failed.
 */
public final class PartialFailureMerger {

    private PartialFailureMerger() {
    }

    /**
     * Merge outcome: the client-facing {@code body}, success/total counts, the absolute item
     * indices that failed, and one reason string per failed chunk (in chunk order — typically
     * homogeneous when an entire FE pool is unhealthy, so callers should de-duplicate before
     * surfacing). The handler returns 200 on any success and reserves 500 for the all-failed
     * case; in either case {@code failedReasons} carries what the FE call returned, including
     * for endpoints whose {@link BatchEndpointSpec.FailedItemFactory} discards reason at the
     * item level (e.g. {@code /batch_infer}'s {@code NULL} factory).
     */
    public record MergedResponse(ObjectNode body,
                                 int succeededChunks,
                                 int totalChunks,
                                 List<Integer> failedIndices,
                                 List<String> failedReasons) {
        public boolean allFailed() {
            return totalChunks > 0 && succeededChunks == 0;
        }
    }

    public static MergedResponse merge(List<SubBatchResult> subs, BatchEndpointSpec spec, ObjectMapper mapper) {
        ObjectNode envelope = null;
        int totalItems = 0;
        for (SubBatchResult s : subs) {
            totalItems += s.chunkSize();
            if (envelope == null && wellFormed(s, spec)) {
                envelope = ((ObjectNode) s.body()).deepCopy();
                envelope.set(spec.getResponseArrayField(), mapper.createArrayNode());
            }
        }
        if (envelope == null) {
            List<String> reasons = new ArrayList<>(subs.size());
            for (SubBatchResult s : subs) {
                reasons.add(reasonFor(s));
            }
            return new MergedResponse(mapper.createObjectNode(), 0, subs.size(),
                    allIndices(totalItems), reasons);
        }
        ArrayNode merged = (ArrayNode) envelope.get(spec.getResponseArrayField());
        List<Integer> failedIndices = new ArrayList<>();
        List<String> failedReasons = new ArrayList<>();
        int succeededChunks = 0;
        for (SubBatchResult s : subs) {
            if (wellFormed(s, spec)) {
                s.body().get(spec.getResponseArrayField()).forEach(merged::add);
                succeededChunks++;
            } else {
                String reason = reasonFor(s);
                failedReasons.add(reason);
                for (int i = 0; i < s.chunkSize(); i++) {
                    int abs = s.startIndex() + i;
                    merged.add(spec.getFailedItemFactory().build(abs, reason, mapper));
                    failedIndices.add(abs);
                }
            }
        }
        if (!failedIndices.isEmpty()) {
            ObjectNode pf = envelope.putObject("_partial_failure");
            pf.put("failed_count", failedIndices.size());
            pf.put("total_count", totalItems);
            ArrayNode fi = pf.putArray("failed_indices");
            failedIndices.forEach(fi::add);
        }
        if (spec.getPostMerger() != null) {
            spec.getPostMerger().apply(envelope, subs, failedIndices, mapper);
        }
        return new MergedResponse(envelope, succeededChunks, subs.size(), failedIndices, failedReasons);
    }

    private static String reasonFor(SubBatchResult s) {
        return s.isSuccess() ? "malformed_sub_batch" : s.reason();
    }

    private static boolean wellFormed(SubBatchResult s, BatchEndpointSpec spec) {
        if (!s.isSuccess() || !(s.body() instanceof ObjectNode on)) {
            return false;
        }
        JsonNode arr = on.get(spec.getResponseArrayField());
        return arr != null && arr.isArray() && arr.size() == s.chunkSize();
    }

    private static List<Integer> allIndices(int n) {
        List<Integer> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(i);
        }
        return out;
    }
}
