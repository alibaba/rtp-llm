package org.flexlb.engine.grpc;

import com.google.protobuf.MapField;
import java.util.Map;
import org.flexlb.cache.primitive.PackedBooleanCacheMap;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PackedCacheMapCompatibilityTest {
    @Test
    void legacyMapPreservesFalseExtremeKeysAndBuilderIsolation() throws Exception {
        var builder = EngineRpcService.CacheStatusPB.newBuilder();
        for (long key : new long[]{0, -1, Long.MIN_VALUE, Long.MAX_VALUE}) {
            builder.putCacheKeys(key, false);
        }
        builder.putCacheKeys(17, true);
        var message = builder.build();
        var parsed = EngineRpcService.CacheStatusPB.parseFrom(message.toByteArray());
        assertEquals(message, parsed);
        assertEquals(message.hashCode(), parsed.hashCode());
        assertFalse(parsed.getCacheKeysOrThrow(0));
        assertFalse(parsed.getCacheKeysMap().containsKey(18L));
        assertNull(parsed.getCacheKeysMap().get(18L));
        assertThrows(UnsupportedOperationException.class,
                () -> parsed.getCacheKeysMap().put(18L, true));
        builder.removeCacheKeys(0).putCacheKeys(17, false);
        assertTrue(message.getCacheKeysOrThrow(17));
        assertEquals(5, message.getCacheKeysCount());
        assertEquals(4, builder.getCacheKeysCount());
        assertEquals(parsed, parsed.toBuilder().build());
        var mapField = parsed.getClass().getDeclaredField("cacheKeys_");
        mapField.setAccessible(true);
        var storage = MapField.class.getDeclaredField("mapData");
        storage.setAccessible(true);
        for (var snapshot : new EngineRpcService.CacheStatusPB[]{parsed, builder.build()}) {
            Object wrapper = storage.get(mapField.get(snapshot));
            var delegate = wrapper.getClass().getDeclaredField("delegate");
            delegate.setAccessible(true);
            assertInstanceOf(PackedBooleanCacheMap.class, delegate.get(wrapper));
        }
    }

    @Test
    void packedListAndLegacyMapCanCoexistOnWire() throws Exception {
        var message = EngineRpcService.CacheStatusPB.newBuilder()
                .putCacheKeys(0, false).addCacheKeyList(Long.MIN_VALUE).build();
        var parsed = EngineRpcService.CacheStatusPB.parseFrom(message.toByteArray());
        assertEquals(Map.of(0L, false), parsed.getCacheKeysMap());
        assertEquals(Long.MIN_VALUE, parsed.getCacheKeyList(0));
    }
}
