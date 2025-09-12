package org.flexlb.domain.batch;

import lombok.Data;

@Data
public class RequestCacheMeta {
    private long sequenceLen;
    private long requestId;
    private long estimateTTFT;
    public RequestCacheMeta(long reqId, long seqLen, long estimateTTFT) {
        this.requestId = reqId;
        this.sequenceLen = seqLen;
        this.estimateTTFT = estimateTTFT;
    }
}
