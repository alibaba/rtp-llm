package org.flexlb.mock.cache;

import java.util.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import static org.flexlb.mock.cache.KvCache.EvictionPolicy.*;
import static org.flexlb.mock.cache.BlockPool.Kind.*;

class TierEvictionTest {
    @Test void hostPruneRemovesDeviceDescendantsButPreservesTheirRequestReferences() {
        var device = new BlockPool(3); var host = new BlockPool(2);
        var tree = new BlockTree(LRU, FIFO);
        var a = tree.insert(1, null, device.allocate(1, CACHE).getFirst(), false);
        var b = tree.insert(2, a, host.allocate(1, CACHE).getFirst(), true);
        var c = tree.insert(3, b, device.allocate(1, CACHE).getFirst(), false);
        var request = device.retain(c.cache, REQUEST);
        assertTrue(tree.evict(true));
        assertEquals(Set.of(1L), tree.keys());
        assertEquals(2, host.snapshot().free());
        assertEquals(1, device.snapshot().free());
        assertEquals(0, device.cachedActiveBlocks());
        request.close(); assertEquals(2, device.snapshot().free()); tree.clear();
        assertEquals(3, device.snapshot().free());
    }
    @Test void deviceReclaimPrunesItsHostSuffixWithoutTouchingOtherHostRoots() {
        var device = new BlockPool(1); var host = new BlockPool(3);
        var tree = new BlockTree(LRU, FIFO);
        tree.insert(9, null, host.allocate(1, CACHE).getFirst(), true);
        var a = tree.insert(1, null, device.allocate(1, CACHE).getFirst(), false);
        tree.insert(2, a, host.allocate(1, CACHE).getFirst(), true);
        assertTrue(tree.evict(false));
        assertEquals(Set.of(9L), tree.keys());
        assertEquals(1, device.snapshot().free());
        assertEquals(2, host.snapshot().free()); tree.clear();
    }
    @Test void tierEntryRefreshesFifoAdmissionButPreservesPopularity() {
        var device = new BlockPool(2); var host = new BlockPool(2);
        var tree = new BlockTree(FIFO, FIFO);
        var first = tree.insert(1, null, device.allocate(1, CACHE).getFirst(), false);
        var second = tree.insert(2, null, host.allocate(1, CACHE).getFirst(), true);
        tree.match(List.of(1L)); long hits = first.hits, access = first.access;
        tree.relocate(first, host.allocate(1, CACHE).getFirst(), true);
        assertEquals(hits, first.hits); assertEquals(access, first.access);
        assertTrue(tree.evict(true)); assertEquals(Set.of(1L), tree.keys());
        var newer = tree.insert(3, null, device.allocate(1, CACHE).getFirst(), false);
        tree.relocate(first, device.allocate(1, CACHE).getFirst(), false);
        assertSame(newer, tree.movableDevice()); tree.clear();
    }
    @Test void oneMatchRefreshesTheWholePrefixWithOneAccessSequence() {
        var pool = new BlockPool(2); var tree = new BlockTree(LRU, FIFO);
        var a = tree.insert(1, null, pool.allocate(1, CACHE).getFirst(), false);
        var b = tree.insert(2, a, pool.allocate(1, CACHE).getFirst(), false);
        tree.match(List.of(1L, 2L));
        assertEquals(a.access, b.access); assertEquals(1, a.hits); assertEquals(1, b.hits);
        tree.clear();
    }
}
