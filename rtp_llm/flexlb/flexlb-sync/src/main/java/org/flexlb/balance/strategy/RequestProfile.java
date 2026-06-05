package org.flexlb.balance.strategy;

public record RequestProfile(long inputLen, long hitCacheTokens) {

    public long computeTokens() {
        return Math.max(0, inputLen - hitCacheTokens);
    }
}
