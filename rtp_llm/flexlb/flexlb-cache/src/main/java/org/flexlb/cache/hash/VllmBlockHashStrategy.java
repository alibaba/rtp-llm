package org.flexlb.cache.hash;

import com.fasterxml.jackson.dataformat.cbor.CBORFactory;
import com.fasterxml.jackson.dataformat.cbor.CBORGenerator;
import org.flexlb.config.BlockHashConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;

import java.io.IOException;
import java.io.OutputStream;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Calculates cache block hashes compatible with vLLM configured with
 * {@code prefix_caching_hash_algo=sha256_cbor}.
 *
 * <p>Sources: {@code vllm/v1/core/kv_cache_utils.py} ({@code init_none_hash},
 * {@code hash_block_tokens}) and {@code vllm/utils/hashing.py} ({@code sha256_cbor}).
 */
public final class VllmBlockHashStrategy implements BlockHashStrategy {

    private static final String DEFAULT_HASH_SEED = "0";
    // vLLM's canonical CBOR uses the shortest representation for every integer.
    private static final CBORFactory CBOR_FACTORY =
            new CBORFactory().enable(CBORGenerator.Feature.WRITE_MINIMAL_INTS);
    private static final ThreadLocal<MessageDigest> SHA_256 =
            ThreadLocal.withInitial(VllmBlockHashStrategy::newSha256Digest);

    private volatile byte[] noneHash;

    /**
     * Creates a strategy with the default vLLM hash seed.
     */
    public VllmBlockHashStrategy() {
        updateHashSeed(DEFAULT_HASH_SEED);
    }

    /**
     * Creates a strategy whose vLLM hash seed follows the supplied configuration service.
     *
     * <p>Registration immediately applies the service's current configuration and keeps the
     * listener registered for subsequent configuration updates during the service's lifetime.
     *
     * @param configService source of the initial and updated vLLM hash seed
     */
    public VllmBlockHashStrategy(ConfigService configService) {
        this();
        configService.addUpdateListener(config -> updateHashSeed(hashSeed(config)));
    }

    @Override
    public List<Long> calculate(int[] inputIds, long blockSize, int lookaheadTokens) {
        if (inputIds == null) {
            throw new IllegalArgumentException("input_ids must not be null");
        }
        if (blockSize <= 0) {
            throw new IllegalArgumentException("block_size must be greater than 0");
        }
        if (lookaheadTokens < 0) {
            throw new IllegalArgumentException("block_hash_lookahead_tokens must not be negative");
        }
        if (inputIds.length == 0 || blockSize > inputIds.length) {
            return Collections.emptyList();
        }

        return calculateBlockCacheKeys(inputIds, (int) blockSize, lookaheadTokens);
    }

    private List<Long> calculateBlockCacheKeys(
            int[] inputIds,
            int blockSize,
            int lookaheadTokens) {
        int fullBlockCount = inputIds.length / blockSize;
        List<Long> blockCacheKeys = new ArrayList<>(fullBlockCount);
        byte[] parentHash = noneHash;
        int tokenIndex = 0;

        MessageDigest digest = SHA_256.get();
        DigestOutputStream digestOutput =
                new DigestOutputStream(OutputStream.nullOutputStream(), digest);
        try (CBORGenerator generator = newCborGenerator(digestOutput)) {
            for (int blockIndex = 0; blockIndex < fullBlockCount; blockIndex++) {
                digest.reset();
                int remainingTokens = inputIds.length - tokenIndex - blockSize;
                int tokenCount = blockSize + Math.min(lookaheadTokens, remainingTokens);
                writeBlock(generator, parentHash, inputIds, tokenIndex, tokenCount);
                generator.flush();

                parentHash = digest.digest();
                blockCacheKeys.add(low64Bits(parentHash));
                tokenIndex += blockSize;
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to encode vLLM block hash input as CBOR", e);
        }
        return blockCacheKeys;
    }

    private void updateHashSeed(String hashSeed) {
        noneHash = calculateNoneHash(hashSeed == null ? DEFAULT_HASH_SEED : hashSeed);
    }

    private static String hashSeed(FlexlbConfig config) {
        BlockHashConfig blockHashConfig = config.getBlockHashConfig();
        return blockHashConfig == null ? DEFAULT_HASH_SEED : blockHashConfig.getHashSeed();
    }

    private static byte[] calculateNoneHash(String hashSeed) {
        MessageDigest digest = SHA_256.get();
        digest.reset();
        DigestOutputStream digestOutput =
                new DigestOutputStream(OutputStream.nullOutputStream(), digest);
        try (CBORGenerator generator = newCborGenerator(digestOutput)) {
            writeHashSeed(generator, hashSeed);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to encode the vLLM hash seed as CBOR", e);
        }
        return digest.digest();
    }

    static CBORGenerator newCborGenerator(OutputStream output) throws IOException {
        return CBOR_FACTORY.createGenerator(output);
    }

    static void writeHashSeed(CBORGenerator generator, String hashSeed) throws IOException {
        generator.writeString(hashSeed);
    }

    static void writeBlock(
            CBORGenerator generator,
            byte[] parentHash,
            int[] inputIds,
            int tokenOffset,
            int blockSize) throws IOException {
        // Supplying array sizes forces definite-length arrays, as required by canonical CBOR.
        generator.writeStartArray(null, 3);
        generator.writeBinary(parentHash);
        generator.writeStartArray(null, blockSize);
        for (int tokenIndex = tokenOffset; tokenIndex < tokenOffset + blockSize; tokenIndex++) {
            generator.writeNumber(inputIds[tokenIndex]);
        }
        generator.writeEndArray();
        generator.writeNull();
        generator.writeEndArray();
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
