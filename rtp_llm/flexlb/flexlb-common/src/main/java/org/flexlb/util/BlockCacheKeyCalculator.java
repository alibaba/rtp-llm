package org.flexlb.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Calculates rolling cache block hashes compatible with RTP-LLM HashUtil.
 * Source: {@code rtp_llm/cpp/utils/HashUtil.h}.
 */
public final class BlockCacheKeyCalculator {

    private static final long HASH_MAGIC = 0x9e3779b97f4a7c15L;

    private BlockCacheKeyCalculator() {
    }

    /**
     * Hashes complete token blocks and drops the final partial block.
     */
    public static List<Long> calculate(List<Long> inputIds, long blockSize) {
        if (inputIds == null) {
            throw new IllegalArgumentException("input_ids must not be null");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("block_size must be greater than 0");
        }
        if (inputIds.isEmpty() || blockSize > inputIds.size()) {
            return Collections.emptyList();
        }

        return calculateBlockCacheKeys(inputIds, (int) blockSize);
    }

    private static List<Long> calculateBlockCacheKeys(List<Long> inputIds, int blockSize) {
        int fullBlockCount = inputIds.size() / blockSize;
        List<Long> blockCacheKeys = new ArrayList<>(fullBlockCount);
        long hash = 0;
        int tokenIndex = 0;

        for (int blockIndex = 0; blockIndex < fullBlockCount; blockIndex++) {
            int blockEnd = tokenIndex + blockSize;
            while (tokenIndex < blockEnd) {
                Long tokenId = inputIds.get(tokenIndex++);
                if (tokenId == null) {
                    throw new IllegalArgumentException("input_ids must not contain null");
                }
                hash = combine(hash, tokenId);
            }
            blockCacheKeys.add(hash);
        }
        return blockCacheKeys;
    }

    private static long combine(long hash, long tokenId) {
        // C++ casts every int64 token ID to int32 before std::hash<int32_t>.
        long tokenHash = (int) tokenId;
        return hash ^ (tokenHash + HASH_MAGIC + (hash << 12) + (hash >> 32));
    }
}
