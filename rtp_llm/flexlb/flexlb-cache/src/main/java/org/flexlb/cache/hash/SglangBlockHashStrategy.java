package org.flexlb.cache.hash;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public final class SglangBlockHashStrategy implements BlockHashStrategy {

    private static final ThreadLocal<MessageDigest> SHA_256 =
            ThreadLocal.withInitial(SglangBlockHashStrategy::newSha256Digest);

    @Override
    public List<Long> calculate(int[] inputIds, long blockSize, int lookaheadTokens) {
        if (inputIds == null) {
            throw new IllegalArgumentException("input_ids must not be null");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("block_size must be greater than 0");
        }
        if (lookaheadTokens < 0 || lookaheadTokens > 1) {
            throw new IllegalArgumentException("SGLang block hashing supports only 0 or 1 lookahead token");
        }
        int logicalLength = inputIds.length - lookaheadTokens;
        if (logicalLength <= 0) {
            return Collections.emptyList();
        }

        int pageSize = (int) blockSize;
        int blockCount = logicalLength / pageSize;
        List<Long> blockCacheKeys = new ArrayList<>(blockCount);
        MessageDigest digest = SHA_256.get();
        byte[] parentHash = null;

        for (int tokenOffset = 0; tokenOffset + pageSize <= logicalLength; tokenOffset += pageSize) {
            digest.reset();
            if (parentHash != null) {
                digest.update(parentHash);
            }
            int tokenEnd = tokenOffset + pageSize;
            for (int tokenIndex = tokenOffset; tokenIndex < tokenEnd; tokenIndex++) {
                updateLittleEndianInt(digest, inputIds[tokenIndex]);
                if (lookaheadTokens == 1) {
                    updateLittleEndianInt(digest, inputIds[tokenIndex + 1]);
                }
            }
            parentHash = digest.digest();
            blockCacheKeys.add(high64Bits(parentHash));
        }
        return blockCacheKeys;
    }

    @Override
    public List<Long> cacheablePrefix(
            List<Long> blockCacheKeys, int inputTokenCount, long blockSize, int lookaheadTokens) {
        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            return Collections.emptyList();
        }
        if (inputTokenCount <= 0 || blockSize <= 0) {
            return blockCacheKeys;
        }
        int logicalLength = Math.max(0, inputTokenCount - lookaheadTokens);
        int fullPageCount = (int) Math.min(blockCacheKeys.size(), logicalLength / blockSize);
        return List.copyOf(blockCacheKeys.subList(0, fullPageCount));
    }

    private static void updateLittleEndianInt(MessageDigest digest, int value) {
        digest.update((byte) value);
        digest.update((byte) (value >>> Byte.SIZE));
        digest.update((byte) (value >>> (2 * Byte.SIZE)));
        digest.update((byte) (value >>> (3 * Byte.SIZE)));
    }

    private static long high64Bits(byte[] hash) {
        long value = 0;
        for (int index = 0; index < Long.BYTES; index++) {
            value = (value << Byte.SIZE) | (hash[index] & 0xffL);
        }
        return value;
    }

    private static MessageDigest newSha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is unavailable", e);
        }
    }
}
