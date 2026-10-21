package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import java.util.ArrayList;
import java.util.List;

/** Frozen release HashUtil + CompleteTokenIds image identity, before page-RR sampling. */
final class LegacyImageCacheKeys {
    private LegacyImageCacheKeys() {}
    static long combine(long hash, int word) {
        return hash ^ ((long)word + 0x9e3779b97f4a7c15L + (hash << 12) + (hash >> 32));
    }
    static long seed() {
        boolean bounded = enabled(System.getenv("DSV41_SWA_BOUNDED_REPLAY"));
        if (bounded && !enabled(System.getenv("DSV41_CED")))
            throw new IllegalArgumentException("DSV41_SWA_BOUNDED_REPLAY requires DSV41_CED=1");
        return bounded ? 0x4453563431525031L : 0;
    }
    private static boolean enabled(String value) {
        return value != null && List.of("1", "true", "yes", "on").contains(value.toLowerCase(java.util.Locale.ROOT));
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
        for (int begin = 0; begin + blockSize <= input.getTokenIdsCount(); begin += blockSize) {
            for (int i = begin; i < begin + blockSize; i++) hash = combine(hash, input.getTokenIds(i));
            if (input.hasV41Inputs()) for (var image : input.getV41Inputs().getImagesList()) {
                int start = image.getStart(), end = Math.addExact(start, image.getTypesCount());
                if (end <= begin || start >= begin + blockSize) continue;
                for (int value : new int[]{-41, start, end, image.getNVitH(), image.getNVitW()}) hash = combine(hash, value);
                for (String digest : List.of(image.getContentSha256(), image.getProcessorIdentity()))
                    for (int offset = 0; offset + 8 <= digest.length(); offset += 8)
                        hash = combine(hash, (int)Long.parseUnsignedLong(digest.substring(offset, offset + 8), 16));
            }
            keys.add(hash);
        }
        return List.copyOf(keys);
    }
    static int validHitBlocks(EngineRpcService.GenerateInputPB input, int hits, int blockSize) {
        hits = Math.min(hits, Math.max(0, (input.getTokenIdsCount() - 1) / blockSize));
        if (input.hasV41Inputs()) for (var image : input.getV41Inputs().getImagesList()) {
            int prefix = hits * blockSize;
            if (prefix > image.getStart() && prefix < image.getStart() + image.getTypesCount())
                hits = image.getStart() / blockSize;
        }
        return hits;
    }
}
