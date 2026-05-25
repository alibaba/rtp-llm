package org.flexlb.dispatcher;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.util.List;

@FunctionalInterface
public interface PostMerger {
    /** Called once after PartialFailureMerger has stitched the array; aggregates cross-chunk fields. */
    void apply(ObjectNode mergedBody, List<SubBatchResult> subs, List<Integer> failedIndices, ObjectMapper mapper);
}
