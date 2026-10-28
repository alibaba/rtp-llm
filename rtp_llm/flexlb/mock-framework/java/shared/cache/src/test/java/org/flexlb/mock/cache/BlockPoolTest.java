package org.flexlb.mock.cache;

import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import static org.flexlb.mock.cache.BlockPool.Kind.*;
import static org.junit.jupiter.api.Assertions.*;

class BlockPoolTest {
    @Test void onlyCacheReferencesAreReclaimableAndEachOwnerClosesIndependently() {
        var pool = new BlockPool(4);
        var request = pool.allocate(2, REQUEST);
        var cached = request.stream().map(r -> pool.retain(r, CACHE)).toList();
        var second = pool.retain(cached.getFirst(), REQUEST);
        request.forEach(BlockPool.Reference::close);
        request.forEach(BlockPool.Reference::close);
        assertEquals(new BlockPool.Snapshot(4, 2, 3, 1), pool.snapshot());
        second.close();
        assertEquals(new BlockPool.Snapshot(4, 2, 4, 0), pool.snapshot());
        var transfer = pool.retain(cached.getFirst(), LOAD);
        cached.forEach(BlockPool.Reference::close);
        assertEquals(new BlockPool.Snapshot(4, 3, 3, 1), pool.snapshot());
        transfer.close();
        assertEquals(new BlockPool.Snapshot(4, 4, 4, 0), pool.snapshot());
        assertThrows(IllegalArgumentException.class, () -> pool.retain(transfer, REQUEST));
    }
    @Test void deviceAndHostUseTheSameCodeWithIndependentCapacities() {
        var device = new BlockPool(2); var host = new BlockPool(3);
        var source = device.allocate(2, CACHE);
        assertNull(device.allocate(1, REQUEST));
        assertThrows(IllegalArgumentException.class, () -> host.retain(source.getFirst(), LOAD));
        var target = host.allocate(2, STORE);
        assertEquals(1, host.snapshot().available());
        source.forEach(BlockPool.Reference::close);
        assertEquals(2, device.snapshot().free());
        assertEquals(1, host.snapshot().free());
        target.forEach(BlockPool.Reference::close);
    }
    @Test void oversizedAllocationDoesNotPartiallyConsumeCapacityAndReleaseWakesWaiter() throws Exception {
        var pool = new BlockPool(2);
        assertNull(pool.allocate(Integer.MAX_VALUE, REQUEST));
        var owned = pool.allocate(2, REQUEST);
        long generation = pool.generation();
        try (var threads = Executors.newVirtualThreadPerTaskExecutor()) {
            var waiter = threads.submit(() -> { pool.awaitChange(generation, 10000); return pool.generation(); });
            owned.getFirst().close();
            assertTrue(waiter.get(1, TimeUnit.SECONDS) > generation);
        }
        owned.forEach(BlockPool.Reference::close);
        assertEquals(2, pool.snapshot().free());
    }
    private record Owner(BlockPool.Reference reference, int block, BlockPool.Kind kind) {}
    @Test void accountingMatchesAnIndependentOwnershipModelDuringRandomRetainAndRelease() {
        var pool = new BlockPool(32);
        var owners = new ArrayList<Owner>(); var random = new Random(28);
        int next = 0;
        for (int i = 0; i < 5000; i++) {
            int action = random.nextInt(3);
            var kind = BlockPool.Kind.values()[random.nextInt(BlockPool.Kind.values().length)];
            if (action == 0 || owners.isEmpty()) {
                var allocation = pool.allocate(1, kind);
                if (allocation != null) owners.add(new Owner(allocation.getFirst(), next++, kind));
            } else if (action == 1) {
                var source = owners.get(random.nextInt(owners.size()));
                owners.add(new Owner(pool.retain(source.reference, kind), source.block, kind));
            } else {
                var removed = owners.remove(random.nextInt(owners.size()));
                removed.reference.close(); removed.reference.close();
            }
            long allocated = owners.stream().map(Owner::block).distinct().count();
            long active = owners.stream().filter(o -> o.kind != CACHE).map(Owner::block).distinct().count();
            assertEquals(new BlockPool.Snapshot(32, 32 - (int) allocated, 32 - (int) active, (int) active), pool.snapshot());
            for (var refKind : BlockPool.Kind.values())
                assertEquals(owners.stream().filter(o -> o.kind == refKind).map(Owner::block).distinct().count(), pool.referencedBlocks(refKind));
        }
        owners.forEach(o -> o.reference.close());
        assertEquals(32, pool.snapshot().free());
    }
}
