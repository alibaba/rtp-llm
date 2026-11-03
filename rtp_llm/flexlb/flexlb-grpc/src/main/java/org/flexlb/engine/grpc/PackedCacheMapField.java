package org.flexlb.engine.grpc;

import com.google.protobuf.MapEntry;
import com.google.protobuf.MapField;
import java.lang.reflect.Field;
import org.flexlb.cache.primitive.PackedBooleanCacheMap;

/** Storage adapter for the pinned protobuf 3.25.1 runtime. */
public final class PackedCacheMapField {
    private static final Field MAP_DATA;
    private static final Field DELEGATE;

    static {
        try {
            MAP_DATA = MapField.class.getDeclaredField("mapData");
            MAP_DATA.setAccessible(true);
            Class<?> wrapper = Class.forName("com.google.protobuf.MapField$MutabilityAwareMap");
            DELEGATE = wrapper.getDeclaredField("delegate");
            DELEGATE.setAccessible(true);
        } catch (ReflectiveOperationException error) {
            throw new ExceptionInInitializerError(error);
        }
    }

    private PackedCacheMapField() {}

    public static MapField<Long, Boolean> copy(MapField<Long, Boolean> source,
                                                MapEntry<Long, Boolean> defaultEntry) {
        MapField<Long, Boolean> copy = create(defaultEntry);
        copy.getMutableMap().putAll(source.getMap());
        return copy;
    }

    public static MapField<Long, Boolean> create(MapEntry<Long, Boolean> defaultEntry) {
        MapField<Long, Boolean> field = MapField.newMapField(defaultEntry);
        try {
            DELEGATE.set(MAP_DATA.get(field), new PackedBooleanCacheMap());
            return field;
        } catch (IllegalAccessException error) {
            throw new IllegalStateException("Unsupported protobuf map storage", error);
        }
    }
}
