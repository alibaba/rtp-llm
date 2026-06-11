package org.flexlb.balance.strategy;

public record BatchRequest(long requestId, long seqLen, long hitCache) {

    public long computeTokens() {
        return Math.max(0, seqLen - hitCache);
    }
}
