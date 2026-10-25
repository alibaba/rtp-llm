package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import java.time.Duration;
import java.util.List;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.*;

class MemoryEvictionIndexTest {
    @Test void consumedHoleStillProtectsItsCommittedAncestors() {
        var c = new MockMemoryBlockCache(4, ignored -> {});
        c.write(List.of(1L, 2L, 3L));
        c.pinRead(List.of(2L), 0).consume();
        c.write(List.of(4L));
        c.write(List.of(5L));
        c.write(List.of(6L));
        assertEquals(Set.of(1L, 4L, 5L, 6L), Set.copyOf(c.keys()));
        c.write(List.of(7L));
        assertFalse(c.keys().contains(1L), "removing the last descendant promotes its ancestor");
    }

    @Test void attachingAFormerRootProtectsTheNewParent() {
        var c = new MockMemoryBlockCache(4, ignored -> {});
        c.write(List.of(2L, 3L));
        c.pinRead(List.of(2L), 0).consume();
        c.write(List.of(1L, 2L));
        c.match(List.of(3L), 0);
        c.write(List.of(4L));
        c.write(List.of(5L));
        assertEquals(Set.of(1L, 2L, 4L, 5L), Set.copyOf(c.keys()));
    }

    @Test void lruTouchesPinsAndOldGenerationLeasesDoNotLeaveStaleVictims() {
        var c = new MockMemoryBlockCache(2, false, ignored -> {}, System::nanoTime);
        c.write(List.of(1L, 2L));
        c.match(List.of(1L), 0);
        c.write(List.of(3L));
        assertEquals(Set.of(1L, 3L), Set.copyOf(c.keys()));
        var lease = c.pinRead(List.of(1L), 0);
        c.clear();
        c.write(List.of(1L, 2L));
        lease.close();
        c.write(List.of(3L));
        assertEquals(Set.of(2L, 3L), Set.copyOf(c.keys()));
        assertEquals(0, c.pinnedBlocks());
    }

    @Test void fullProductionSizedBranchedPoolSupportsSustainedEviction() {
        assertTimeoutPreemptively(Duration.ofSeconds(5), () -> {
            var c = new MockMemoryBlockCache(52_295, ignored -> {});
            for (long k = 2; k <= 52_295; k++) c.write(List.of(1L, k));
            for (long k = 52_296; k <= 57_295; k++) c.write(List.of(1L, k));
            assertEquals(52_295, c.size());
            assertEquals(5_000, c.evictions());
            assertTrue(c.keys().contains(1L));
            assertFalse(c.keys().contains(5_001L));
            assertTrue(c.keys().contains(5_002L));
        });
    }
}
