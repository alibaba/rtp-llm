package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import java.util.List;

/** Native requests follow StreamCacheResource and CoordinatorCacheManager.
 * Explicit replay keys retain their synthetic trace accounting contract.
 */
final class MockCachePolicy {
    private MockCachePolicy() { }
    static boolean reuse(EngineRpcService.GenerateInputPB input, boolean nativeKeys, boolean enabled) {
        return enabled && (!nativeKeys || input.getGenerateConfig().getReuseCache());
    }
    static List<Long> matchKeys(List<Long> keys, int inputLength, int blockSize, boolean nativeKeys) {
        int limit = nativeKeys ? Math.max(0, (inputLength - 1) / blockSize) : keys.size();
        return keys.subList(0, Math.min(keys.size(), limit));
    }
}
