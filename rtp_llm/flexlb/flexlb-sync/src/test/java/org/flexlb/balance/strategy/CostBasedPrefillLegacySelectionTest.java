package org.flexlb.balance.strategy;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

/** Contracts that keep delivery admission out of legacy Prefill selection. */
class CostBasedPrefillLegacySelectionTest {

    @Test
    void decodeAdmissionBlockDoesNotRemovePrefillCandidate() {
        GroupPlanner.Item head = new GroupPlanner.Item(
                1L, 50, 1L, 1_000L, 10_000L, 128L, 0L);
        QueueSnapshot queue = new QueueSnapshot(
                2_000L,
                true,
                Comparator.comparingLong(GroupPlanner.Item::enqueueSeq),
                new GroupPlanner.Constraints(
                        8, 4_096L, 4_096L, 0L, 10L),
                List.of(head),
                new QueueSnapshot.AdmissionBlock(
                        head.requestId(),
                        head.enqueueSeq(),
                        new RouteProjection.AdmissionBlockSemantics(
                                "decode full",
                                RouteProjection.AfterProbeAdmission.BLOCKED,
                                "decode full",
                                RoleType.DECODE)));
        RouteProjection.Inputs captured = new RouteProjection.Inputs(
                queue,
                new WorkSnapshot(
                        queue.capturedAtMs(), List.of(), List.of(), 3L),
                3L);

        RouteProjection.Inputs selection =
                CostBasedPrefillStrategy.legacyProjectionInputs(captured);

        assertNull(selection.queue().admissionBlock());
        assertEquals(0L, selection.work().unknownRequestCount());
        assertEquals(3L, selection.pendingRequestCount(),
                "Engine-only work still participates in the hard pending gate");
    }
}
