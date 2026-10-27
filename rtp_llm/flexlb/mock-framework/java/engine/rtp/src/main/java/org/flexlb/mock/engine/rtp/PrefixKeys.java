package org.flexlb.mock.engine.rtp;

import java.util.*;
import org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB;

/** Text-only native Jenkins rolling keys, matching cpp/utils/HashUtil.h (signed arithmetic shift). */
final class PrefixKeys {
    private PrefixKeys() {}
    static List<Long> completeBlocks(GenerateInputPB input, int blockSize) {
        List<Long> keys = new ArrayList<>();
        long hash = 0;
        for (int i = 0; i < input.getTokenIdsCount(); i++) {
            hash ^= (long) input.getTokenIds(i) + 0x9e3779b97f4a7c15L + (hash << 12) + (hash >> 32);
            if ((i + 1) % blockSize == 0) keys.add(hash);
        }
        return List.copyOf(keys);
    }
}
