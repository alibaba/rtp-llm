package org.flexlb.mock.master;

import static org.junit.jupiter.api.Assertions.*;
import java.util.*;
import java.util.concurrent.atomic.*;
import org.flexlb.mock.api.model.Types.*;
import org.junit.jupiter.api.Test;

class MasterReadsTest {
    @Test void leadershipUsesTheActualClaimIncludingNoElectionInStandaloneMode() {
        var leader = new AtomicBoolean();
        var reads = new MasterReads("master-0", leader::get, role -> Set.of(), Map.of());
        assertTrue(reads.leadership(new None()).instances().isEmpty());
        leader.set(true); assertEquals(List.of(new MasterRef("master-0")), reads.leadership(new None()).instances());
    }
    @Test void membershipComesFromObservedAddressesAndReturnsStableHandlesPerMaster() {
        var observed = new AtomicReference<Set<String>>(Set.of("127.0.0.1:9002", "127.0.0.1:9001"));
        var reads = new MasterReads("master-0", () -> false, role -> role == Role.prefill ? observed.get() : Set.of(),
            Map.of("127.0.0.1:9001", "prefill-0", "127.0.0.1:9002", "prefill-1"));
        assertEquals(List.of(new EngineRef("prefill-0"), new EngineRef("prefill-1")),
            reads.engines(new EngineSelection(Role.prefill)).instances().get("master-0").instances());
        observed.set(Set.of());
        assertTrue(reads.engines(new EngineSelection(Role.prefill)).instances().get("master-0").instances().isEmpty());
        assertTrue(reads.engines(new EngineSelection(Role.decode)).instances().get("master-0").instances().isEmpty());
    }
    @Test void unknownObservedAddressIsAnErrorInsteadOfSilentlyDroppingIt() {
        var reads = new MasterReads("master-0", () -> false, role -> Set.of("unknown:9000"), Map.of());
        assertThrows(IllegalStateException.class, () -> reads.engines(new EngineSelection(Role.prefill)));
    }
}
