package org.flexlb.mockengine;

import java.util.LinkedHashMap;

/** Worker-local URL data cache; real MMDataCache is bounded by items, not embedding bytes. */
final class MockVitMediaCache {
    private final int capacity;
    private final LinkedHashMap<String, byte[]> entries = new LinkedHashMap<>(16, .75f, true);
    MockVitMediaCache(int capacity) { this.capacity = Math.max(0, capacity); }
    synchronized byte[] get(String url) { return entries.get(url); }
    synchronized void put(String url, byte[] data) {
        if (capacity == 0) return;
        entries.put(url, data);
        while (entries.size() > capacity) entries.remove(entries.firstEntry().getKey());
    }
    synchronized long size() { return entries.size(); }
    synchronized void clear() { entries.clear(); }
}
