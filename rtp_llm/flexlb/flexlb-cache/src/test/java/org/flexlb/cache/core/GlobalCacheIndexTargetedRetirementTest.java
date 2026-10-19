package org.flexlb.cache.core;

import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Set;
import static org.junit.jupiter.api.Assertions.*;

class GlobalCacheIndexTargetedRetirementTest {
    @Test void targetedRetirementPreservesForeignOwnersAndUnrelatedKeys() {
        var index=new GlobalCacheIndex();
        index.addCacheBlock(1L,"removed");
        index.addCacheBlock(1L,"retained");
        index.addCacheBlock(2L,"retained");
        index.removeEngineKeys("removed",Set.of(1L));
        index.removeEngineKeys("removed",Set.of(1L));
        assertEquals(2,index.totalBlocks());
        assertEquals(2,index.totalMappings());
        assertEquals(0,index.batchCalculatePrefixMatchLength(List.of("removed"),List.of(1L,2L)).get("removed"));
        assertEquals(2,index.batchCalculatePrefixMatchLength(List.of("retained"),List.of(1L,2L)).get("retained"));
    }
}
