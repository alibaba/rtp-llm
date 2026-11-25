package org.flexlb.balance.strategy;

/** Resident prefix tokens and the tokens saved after accounting for last-block recomputation. */
public record CacheTokenMatch(long effectiveHitTokens, long routingHitTokens) {
    static final CacheTokenMatch NONE = new CacheTokenMatch(0L, 0L);

    public static CacheTokenMatch fromBlocks(long input, long blockSize, long matchedBlocks) {
        if (input <= 0 || blockSize <= 0 || matchedBlocks <= 0) { return NONE; }
        long rawHit;
        try {
            rawHit = Math.multiplyExact(blockSize, matchedBlocks);
        } catch (ArithmeticException overflow) {
            rawHit = input;
        }
        return rawHit >= input
                ? new CacheTokenMatch(Math.max(0L, input - blockSize), input)
                : new CacheTokenMatch(rawHit, rawHit);
    }
}
