package org.flexlb.mockengine;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;

final class MockLruBlockCache {
    private final int capacity;
    private final LinkedHashMap<Long, Boolean> blocks;

    MockLruBlockCache(int capacity) {
        this.capacity = Math.max(0, capacity);
        this.blocks = new LinkedHashMap<>(16, 0.75f, true);
    }

    synchronized int prefixHitBlocks(List<Long> keys) {
        int hits = 0;
        for (Long key : keys) {
            if (!blocks.containsKey(key)) {
                break;
            }
            blocks.get(key);
            hits++;
        }
        return hits;
    }

    synchronized boolean admit(List<Long> keys) {
        if (capacity == 0 || keys.isEmpty()) {
            return false;
        }
        boolean changed = false;
        for (Long key : keys) {
            changed |= blocks.put(key, Boolean.TRUE) == null;
        }
        while (blocks.size() > capacity) {
            Long eldest = blocks.keySet().iterator().next();
            blocks.remove(eldest);
        }
        return changed;
    }

    synchronized Set<Long> snapshotKeys() {
        return Set.copyOf(blocks.keySet());
    }
}
