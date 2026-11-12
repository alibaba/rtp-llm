package org.flexlb.cache.primitive;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;

import java.util.Collection;
import java.util.Set;

/** Primitive key storage. Mutations require the owning generation's cache mutation lock. */
public final class LongKeySet extends LongOpenHashSet {
    public LongKeySet() { super(); }
    public LongKeySet(int expected) { super(expected); }
    public LongKeySet(Collection<? extends Long> keys) { super(keys); }

    public static LongKeySet difference(Set<Long> source, Set<Long> excluded) {
        if (excluded.isEmpty()) {
            return new LongKeySet(source);
        }
        LongKeySet result = new LongKeySet();
        if (source instanceof LongKeySet primitiveSource) {
            var iterator = primitiveSource.iterator();
            while (iterator.hasNext()) {
                long key = iterator.nextLong();
                boolean present = excluded instanceof LongKeySet primitiveExcluded
                        ? primitiveExcluded.contains(key) : excluded.contains(key);
                if (!present) {
                    result.add(key);
                }
            }
        } else {
            for (Long key : source) {
                if (!excluded.contains(key)) {
                    result.add(key.longValue());
                }
            }
        }
        return result;
    }
}
