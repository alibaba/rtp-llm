package org.flexlb.service.grpc;

import java.util.Set;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CacheKeyListCompatibilityTest {
    @Test
    void nonemptyListOverridesMapAndPreservesSignedKeys() throws Exception {
        var message = EngineRpcService.CacheStatusPB.newBuilder()
                .putCacheKeys(42L, true)
                .addCacheKeyList(Long.MIN_VALUE)
                .addCacheKeyList(Long.MAX_VALUE)
                .addCacheKeyList(Long.MIN_VALUE)
                .build();
        var decoded = EngineRpcService.CacheStatusPB.parseFrom(message.toByteArray());
        var status = EngineStatusConverter.convertToCacheStatus(decoded);
        assertEquals(Set.of(Long.MIN_VALUE, Long.MAX_VALUE), status.getCachedKeys());
        assertEquals(2, status.getCacheKeySize());
    }

    @Test
    void emptyListFallsBackToMapRegardlessOfBoolValue() {
        var message = EngineRpcService.CacheStatusPB.newBuilder()
                .putCacheKeys(1L, true).putCacheKeys(2L, false).build();
        assertEquals(Set.of(1L, 2L), EngineStatusConverter.convertToCacheStatus(message).getCachedKeys());
    }

    @Test
    void emptyMessageHasNoKeys() {
        var status = EngineStatusConverter.convertToCacheStatus(
                EngineRpcService.CacheStatusPB.getDefaultInstance());
        assertEquals(Set.of(), status.getCachedKeys());
        assertEquals(0, status.getCacheKeySize());
    }
}
