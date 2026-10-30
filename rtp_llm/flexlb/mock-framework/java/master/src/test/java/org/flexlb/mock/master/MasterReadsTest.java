package org.flexlb.mock.master;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.atomic.*;
import org.flexlb.mock.api.model.Types.*;
import org.junit.jupiter.api.Test;

class MasterReadsTest {
    @Test void leadershipUsesTheActualClaimIncludingBeforeElectionCompletes() {
        var leader = new AtomicBoolean();
        var reads = new MasterReads("master-0", leader::get, role -> Set.of());
        assertTrue(reads.leadership(new None()).instances().isEmpty());
        leader.set(true); assertEquals(List.of(new MasterRef("master-0")), reads.leadership(new None()).instances());
    }
    @Test void membershipReturnsObservedAddressesPerMaster() {
        var observed = new AtomicReference<Set<String>>(Set.of("127.0.0.1:9002", "127.0.0.1:9001"));
        var reads = new MasterReads("master-0", () -> false, role -> role == Role.prefill ? observed.get() : Set.of());
        assertEquals(List.of("127.0.0.1:9001", "127.0.0.1:9002"),
            reads.engines(new EngineSelection(Role.prefill)).instances().get("master-0"));
        observed.set(Set.of());
        assertTrue(reads.engines(new EngineSelection(Role.prefill)).instances().get("master-0").isEmpty());
        assertTrue(reads.engines(new EngineSelection(Role.decode)).instances().get("master-0").isEmpty());
    }
    @Test void retiredOrUnknownObservedAddressIsReturnedWithoutARegistry() {
        var reads = new MasterReads("master-0", () -> false, role -> Set.of("unknown:9000"));
        assertEquals(List.of("unknown:9000"), reads.engines(new EngineSelection(Role.prefill)).instances().get("master-0"));
    }
}
