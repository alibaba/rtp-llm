package org.flexlb.util;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Calculates cache block hashes compatible with vLLM configured with
 * {@code prefix_caching_hash_algo=sha256_cbor} and {@code PYTHONHASHSEED=0}.
 *
 * <p>Sources: {@code vllm/v1/core/kv_cache_utils.py} ({@code init_none_hash},
 * {@code hash_block_tokens}) and {@code vllm/utils/hashing.py} ({@code sha256_cbor}).
 */
public final class BlockCacheKeyCalculator {

    private static final int CBOR_BYTE_STRING = 2;
    private static final int CBOR_TEXT_STRING = 3;
    private static final int CBOR_ARRAY = 4;
    private static final byte CBOR_NULL = (byte) 0xf6;
    private static final byte[] HASH_SEED = "0".getBytes(StandardCharsets.UTF_8);
    private static final ThreadLocal<MessageDigest> SHA_256 =
            ThreadLocal.withInitial(BlockCacheKeyCalculator::newSha256Digest);
    private static final byte[] NONE_HASH = calculateNoneHash();

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
        byte[] parentHash = NONE_HASH;
        int tokenIndex = 0;

        for (int blockIndex = 0; blockIndex < fullBlockCount; blockIndex++) {
            MessageDigest digest = SHA_256.get();
            digest.reset();
            updateTypeAndLength(digest, CBOR_ARRAY, 3);
            updateByteString(digest, parentHash);
            updateTypeAndLength(digest, CBOR_ARRAY, blockSize);

            for (int blockTokenIndex = 0; blockTokenIndex < blockSize; blockTokenIndex++) {
                Long tokenId = inputIds.get(tokenIndex++);
                if (tokenId == null) {
                    throw new IllegalArgumentException("input_ids must not contain null");
                }
                updateInteger(digest, tokenId);
            }
            digest.update(CBOR_NULL);

            parentHash = digest.digest();
            blockCacheKeys.add(low64Bits(parentHash));
        }
        return blockCacheKeys;
    }

    private static byte[] calculateNoneHash() {
        MessageDigest digest = SHA_256.get();
        digest.reset();
        updateTypeAndLength(digest, CBOR_TEXT_STRING, HASH_SEED.length);
        digest.update(HASH_SEED);
        return digest.digest();
    }

    private static void updateByteString(MessageDigest digest, byte[] value) {
        updateTypeAndLength(digest, CBOR_BYTE_STRING, value.length);
        digest.update(value);
    }

    private static void updateInteger(MessageDigest digest, long value) {
        if (value >= 0) {
            updateTypeAndLength(digest, 0, value);
        } else {
            updateTypeAndLength(digest, 1, -1 - value);
        }
    }

    private static void updateTypeAndLength(MessageDigest digest, int majorType, long value) {
        int type = majorType << 5;
        if (value < 24) {
            digest.update((byte) (type | value));
        } else if (value <= 0xffL) {
            digest.update((byte) (type | 24));
            digest.update((byte) value);
        } else if (value <= 0xffffL) {
            digest.update((byte) (type | 25));
            updateBigEndian(digest, value, 2);
        } else if (value <= 0xffff_ffffL) {
            digest.update((byte) (type | 26));
            updateBigEndian(digest, value, 4);
        } else {
            digest.update((byte) (type | 27));
            updateBigEndian(digest, value, 8);
        }
    }

    private static void updateBigEndian(MessageDigest digest, long value, int byteCount) {
        for (int shift = (byteCount - 1) * Byte.SIZE; shift >= 0; shift -= Byte.SIZE) {
            digest.update((byte) (value >>> shift));
        }
    }

    private static long low64Bits(byte[] hash) {
        long value = 0;
        for (int index = hash.length - Long.BYTES; index < hash.length; index++) {
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
