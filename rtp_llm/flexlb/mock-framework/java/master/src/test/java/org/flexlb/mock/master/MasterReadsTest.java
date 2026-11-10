package org.flexlb.mock.master;

import org.flexlb.MasterNode.EngineSelection;
import org.flexlb.MasterNode.MasterReads;
import org.flexlb.MasterNode.MasterRef;
import org.flexlb.MasterNode.None;
import org.flexlb.mock.protocol.Role;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks the Master read endpoints with controlled leadership and worker state.
 */
class MasterReadsTest {
    @Test
    void leadershipUsesTheActualClaimIncludingBeforeElectionCompletes() {
        var leader = new AtomicBoolean();
        var reads = new MasterReads("master-0", leader::get, role -> Set.of());
        assertTrue(reads.leadership(new None()).instances().isEmpty());
        leader.set(true);
        assertEquals(List.of(new MasterRef("master-0")), reads.leadership(new None()).instances());
    }

    @Test
    void membershipReturnsObservedAddressesPerMaster() {
        var observed = new AtomicReference<Set<String>>(Set.of("127.0.0.1:9002", "127.0.0.1:9001"));
        var reads =
                new MasterReads(
                        "master-0",
                        () -> false,
                        role -> role == Role.prefill ? observed.get() : Set.of());
        assertEquals(
                List.of("127.0.0.1:9001", "127.0.0.1:9002"),
                reads.engines(new EngineSelection(Role.prefill)).instances().get("master-0"));
        observed.set(Set.of());
        assertTrue(
                reads.engines(new EngineSelection(Role.prefill))
                        .instances()
                        .get("master-0")
                        .isEmpty());
        assertTrue(
                reads.engines(new EngineSelection(Role.decode))
                        .instances()
                        .get("master-0")
                        .isEmpty());
    }

    @Test
    void retiredOrUnknownObservedAddressIsReturnedWithoutARegistry() {
        var reads = new MasterReads("master-0", () -> false, role -> Set.of("unknown:9000"));
        assertEquals(
                List.of("unknown:9000"),
                reads.engines(new EngineSelection(Role.prefill)).instances().get("master-0"));
    }
}
