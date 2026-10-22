package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import java.util.ArrayList;
import java.util.List;

/** main HashUtil + CompleteTokenIds image identity, before page-RR sampling. */
final class MockImageCacheKeys {
    private MockImageCacheKeys() {}
    static long combine(long hash, int word) {
        return hash ^ ((long)word + 0x9e3779b97f4a7c15L + (hash << 12) + (hash >> 32));
    }
    static long seed(EngineRpcService.GenerateInputPB input) {
        String override = System.getenv("MOCK_CACHE_KEY_HASH_SEED");
        if (override != null && !override.isBlank()) return Long.decode(override);
        if (!input.hasV41Inputs()) return 0;
        boolean bounded = "1".equals(System.getenv("DSV41_SWA_BOUNDED_REPLAY"));
        return bounded ? 0x4453563431494231L : 0x4453563431494631L;
    }
    static List<Long> routeBlocks(EngineRpcService.GenerateInputPB input, int physicalBlockSize,
                                 int routeBlockSize, long initialHash) {
        if (physicalBlockSize <= 0 || routeBlockSize <= 0 || routeBlockSize % physicalBlockSize != 0)
            throw new IllegalArgumentException("role block size must be a multiple of physical hash block size");
        List<Long> physical = fullBlocks(input, physicalBlockSize, initialHash);
        int stride = routeBlockSize / physicalBlockSize;
        List<Long> result = new ArrayList<>();
        for (int i = stride - 1; i < physical.size(); i += stride) result.add(physical.get(i));
        return List.copyOf(result);
    }
    static List<Long> fullBlocks(EngineRpcService.GenerateInputPB input, int blockSize, long initialHash) {
        List<Long> keys = new ArrayList<>();
        long hash = initialHash;
        int trustedLength = input.getTokenIdsCount();
        if (input.hasV41Inputs()) for (var image : input.getV41Inputs().getImagesList())
            if (image.getContentSha256().isEmpty() || image.getProcessorIdentity().isEmpty())
                trustedLength = Math.min(trustedLength, image.getStart());
        for (int begin = 0; begin + blockSize <= trustedLength; begin += blockSize) {
            for (int i = begin; i < begin + blockSize; i++) hash = combine(hash, input.getTokenIds(i));
            if (input.hasV41Inputs()) for (var image : input.getV41Inputs().getImagesList()) {
                int start = image.getStart(), end = Math.addExact(start, image.getTypesCount());
                if (end <= begin || start >= begin + blockSize) continue;
                for (int value : new int[]{-41, start, end, image.getNVitH(), image.getNVitW()}) hash = combine(hash, value);
                for (String digest : List.of(image.getContentSha256(), image.getProcessorIdentity())) {
                    byte[] bytes = digest.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                    hash = combine(hash, bytes.length);
                    for (byte value : bytes) hash = combine(hash, Byte.toUnsignedInt(value));
                }
            }
            keys.add(hash);
        }
        return List.copyOf(keys);
    }
    static int validHitBlocks(EngineRpcService.GenerateInputPB input, int hits, int blockSize) {
        hits = Math.min(hits, Math.max(0, (input.getTokenIdsCount() - 1) / blockSize));
        if (input.hasV41Inputs()) for (var image : input.getV41Inputs().getImagesList()) {
            int prefix = hits * blockSize;
            if (prefix > image.getStart() && (image.getContentSha256().isEmpty() || image.getProcessorIdentity().isEmpty()
                    || prefix < image.getStart() + image.getTypesCount()))
                hits = image.getStart() / blockSize;
        }
        return hits;
    }
}
