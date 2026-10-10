package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class CacheReferenceAccountingTest {
    private record Active(MockLruBlockCache.BlockLease lease, List<Long> keys) {}

    @Test
    void incrementalCountMatchesFullScanAcrossSharedLeasesAndEvictions() throws Exception {
        var cache = new MockLruBlockCache(64);
        var field = MockLruBlockCache.class.getDeclaredField("blocks");
        field.setAccessible(true);
        var random = new Random(20260913);
        var active = new ArrayList<Active>();
        for (int step = 0; step < 5000; step++) {
            int op = random.nextInt(5);
            long start = random.nextInt(100);
            var keys = List.of(start, start + 1, start + 2);
            if (op < 2) {
                var lease = op == 0 ? cache.acquire(3, keys) : cache.acquireWithReuse(3, keys);
                if (lease != null) {
                    if (op == 0 && random.nextBoolean()) lease = cache.retainComputed(lease, keys);
                    active.add(new Active(lease, keys));
                }
            } else if (op == 2 && !active.isEmpty()) {
                var item = active.remove(random.nextInt(active.size()));
                if (random.nextBoolean()) cache.release(item.lease());
                else cache.admit(item.lease(), item.keys());
            } else if (op == 3) {
                cache.evict(keys);
            } else if (active.isEmpty()) {
                cache.clear();
            }
            @SuppressWarnings("unchecked")
            var blocks = (Map<Long,Integer>) field.get(cache);
            int referenced = (int) blocks.values().stream().filter(v -> v > 0).count();
            assertEquals(referenced, cache.referencedKeyBlocks(), "step " + step);
            assertEquals(64 - cache.heldBlocks() - referenced, cache.availableBlocks());
        }
        for (var item : active) cache.release(item.lease());
        assertEquals(0, cache.referencedKeyBlocks());
        assertEquals(64, cache.availableBlocks());
        cache.clear();
        assertEquals(64, cache.freeBlocks());
    }
}
