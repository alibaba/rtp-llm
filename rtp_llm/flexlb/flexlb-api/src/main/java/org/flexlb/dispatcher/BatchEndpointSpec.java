package org.flexlb.dispatcher;

import lombok.Value;

@Value
public class BatchEndpointSpec {
    String path;                           // e.g. "/batch_infer"
    String requestArrayField;              // e.g. "prompt_batch"
    String responseArrayField;             // e.g. "response_batch"
    FailedItemFactory failedItemFactory;
    /** May be null when an endpoint has no cross-chunk aggregation. */
    PostMerger postMerger;
}
