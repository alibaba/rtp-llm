package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.OptionalLong;

import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.ROUTE;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.TOKEN_EVALUATOR;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.constraints;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.item;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.noCommittedWork;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.probe;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.project;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.work;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

/** Work prediction and observed-head admission must remain independent. */
class RouteAdmissionPolicyTest {

    @Test
    void decodeWaitDoesNotEraseKnownWorkPrediction() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueSnapshot queue =
                blockedQueue(
                        false, head, semantics(RouteProjection.AfterProbeAdmission.INDEPENDENT));
        RouteProjection.Candidate result =
                project(
                        queue,
                        work(List.of(), List.of(), 1L),
                        TOKEN_EVALUATOR,
                        probe(99L, 50, 20L, 0L),
                        ROUTE);

        assertEquals(OptionalLong.of(120L), result.projectedTtftMs());
        assertEquals(
                RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
        assertNull(admissionBlockReason(queue, result, 50));
        assertEquals(OptionalLong.of(120L), result.projectedTtftMs());
    }

    @Test
    void deliveryOnlyWaitPreservesTheQueueForecast() {
        QueueSnapshot queue = blockedQueue(false, item(1L, 50, 1L, 100L), null);
        RouteProjection.Candidate result =
                project(queue, noCommittedWork(), TOKEN_EVALUATOR, probe(99L, 50, 20L, 0L), ROUTE);

        assertEquals(OptionalLong.of(120L), result.projectedTtftMs());
        assertEquals(
                RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
        assertNull(admissionBlockReason(queue, result, 50));
    }

    @ParameterizedTest
    @ValueSource(longs = {0L, 1L})
    void fifoDecodeBlockedHeadAllowsHigherPriorityProbe(long unknownRequestCount) {
        QueueSnapshot queue =
                blockedQueue(
                        false,
                        item(1L, 50, 1L, 100L),
                        semantics(RouteProjection.AfterProbeAdmission.INDEPENDENT));
        RouteProjection.Candidate result =
                project(
                        queue,
                        work(List.of(), List.of(), unknownRequestCount),
                        TOKEN_EVALUATOR,
                        probe(99L, 100, 20L, 0L),
                        ROUTE);

        assertEquals(OptionalLong.of(120L), result.projectedTtftMs());
        assertNull(admissionBlockReason(queue, result, 100));
    }

    @Test
    void prioritySameOrLowerProbeRemainsBehindObservedHead() {
        QueueSnapshot queue =
                blockedQueue(
                        true,
                        item(1L, 50, 1L, 100L),
                        semantics(RouteProjection.AfterProbeAdmission.BLOCKED));
        for (int priority : List.of(50, 49)) {
            RouteProjection.Candidate result =
                    project(
                            queue,
                            noCommittedWork(),
                            TOKEN_EVALUATOR,
                            probe(99L, priority, 20L, 0L),
                            ROUTE);
            assertEquals(OptionalLong.of(120L), result.projectedTtftMs());
            assertEquals(
                    RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                    result.initialHeadDisposition());
            assertEquals("HEAD_CAPACITY_BLOCKED", admissionBlockReason(queue, result, priority));
        }
    }

    @Test
    void higherPriorityProbeCanRemainHardBlockedByCapturedSemantics() {
        QueueSnapshot queue =
                blockedQueue(
                        true,
                        item(1L, 50, 1L, 100L),
                        semantics(RouteProjection.AfterProbeAdmission.BLOCKED));
        RouteProjection.Candidate result =
                project(queue, noCommittedWork(), TOKEN_EVALUATOR, probe(99L, 90, 20L, 0L), ROUTE);

        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", admissionBlockReason(queue, result, 90));
    }

    @ParameterizedTest
    @ValueSource(longs = {0L, 1L})
    void higherPriorityProbeDoesNotInheritAnotherRequestsDecodeBlock(long unknownRequestCount) {
        QueueSnapshot queue =
                blockedQueue(
                        true,
                        item(1L, 50, 1L, 100L),
                        semantics(RouteProjection.AfterProbeAdmission.INDEPENDENT));
        RouteProjection.Candidate result =
                project(
                        queue,
                        work(List.of(), List.of(), unknownRequestCount),
                        TOKEN_EVALUATOR,
                        probe(99L, 90, 20L, 0L),
                        ROUTE);

        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(
                RouteProjection.Candidate.InitialHeadDisposition.AFTER_PROBE,
                result.initialHeadDisposition());
        assertNull(admissionBlockReason(queue, result, 90));
    }

    @Test
    void decodeCapacityEvidenceSurvivesAnObservedHeadAheadOfProbe() {
        QueueSnapshot queue =
                blockedQueue(
                        true,
                        item(1L, 50, 1L, 100L),
                        semantics(RouteProjection.AfterProbeAdmission.INDEPENDENT));
        RouteProjection.Candidate result =
                project(queue, noCommittedWork(), TOKEN_EVALUATOR, probe(99L, 40, 20L, 0L), ROUTE);

        assertEquals(OptionalLong.of(120L), result.projectedTtftMs());
        assertEquals(
                RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
        assertEquals("HEAD_CAPACITY_BLOCKED", admissionBlockReason(queue, result, 40));
    }

    @Test
    void unknownWorkDoesNotBypassSharedCapacityBlock() {
        QueueSnapshot queue =
                blockedQueue(
                        true,
                        item(1L, 50, 1L, 100L),
                        semantics(RouteProjection.AfterProbeAdmission.BLOCKED));
        RouteProjection.Candidate result =
                project(
                        queue,
                        work(List.of(), List.of(), 1L),
                        TOKEN_EVALUATOR,
                        probe(99L, 90, 20L, 0L),
                        ROUTE);

        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(
                RouteProjection.Candidate.InitialHeadDisposition.AFTER_PROBE,
                result.initialHeadDisposition());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", admissionBlockReason(queue, result, 90));
    }

    @ParameterizedTest
    @ValueSource(longs = {0L, 1L})
    void terminallyPrunedObservedHeadDoesNotBlockProbe(long unknownRequestCount) {
        GroupPlanner.Item expired = item(1L, 50, 1L, 100L, RouteProjectionTestSupport.NOW_MS);
        QueueSnapshot queue =
                blockedQueue(
                        false, expired, semantics(RouteProjection.AfterProbeAdmission.BLOCKED));
        RouteProjection.Candidate result =
                project(
                        queue,
                        work(List.of(), List.of(), unknownRequestCount),
                        TOKEN_EVALUATOR,
                        probe(99L, 50, 20L, 0L),
                        ROUTE);

        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(
                RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED,
                result.initialHeadDisposition());
        assertNull(admissionBlockReason(queue, result, 50));
    }

    @Test
    void directProjectionIgnoresPriorQueueAdmissionObservation() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueSnapshot queue =
                RouteProjectionTestSupport.queue(
                        false,
                        false,
                        constraints(1, 0L),
                        List.of(head),
                        block(head, semantics(RouteProjection.AfterProbeAdmission.BLOCKED)));
        RouteProjection.Candidate result =
                project(queue, noCommittedWork(), TOKEN_EVALUATOR, probe(99L, 50, 20L, 0L), ROUTE);

        assertEquals(OptionalLong.of(20L), result.projectedTtftMs());
        assertEquals(
                RouteProjection.Candidate.InitialHeadDisposition.NONE,
                result.initialHeadDisposition());
        assertNull(admissionBlockReason(queue, result, 50));
    }

    @Test
    void predictionFailureKeepsItsOwnReason() {
        QueueSnapshot queue =
                blockedQueue(
                        false,
                        item(1L, 50, 1L, 100L),
                        semantics(RouteProjection.AfterProbeAdmission.INDEPENDENT));
        RouteProjection.Candidate result =
                project(queue, noCommittedWork(), null, probe(99L, 50, 20L, 0L), ROUTE);

        assertFalse(result.hasPrediction());
        assertEquals("PREDICTOR_MISSING", result.detail());
        assertNull(admissionBlockReason(queue, result, 50));
    }

    @Test
    void blockedSemanticsCannotOmitTheWaitReason() {
        assertThrows(
                NullPointerException.class,
                () ->
                        new RouteProjection.AdmissionBlockSemantics(
                                null, RouteProjection.AfterProbeAdmission.INDEPENDENT, ""));
        assertThrows(
                NullPointerException.class,
                () ->
                        new RouteProjection.AdmissionBlockSemantics(
                                "HEAD", RouteProjection.AfterProbeAdmission.BLOCKED, null));
        assertThrows(
                NullPointerException.class,
                () -> new RouteProjection.AdmissionBlockSemantics("HEAD", null, ""));
    }

    private static String admissionBlockReason(
            QueueSnapshot queue, RouteProjection.Candidate result, int priority) {
        return RouteProjection.admissionBlockReason(
                queue,
                result,
                item(99L, priority, Long.MAX_VALUE, 20L),
                RouteProjectionTestSupport.NOW_MS);
    }

    private static QueueSnapshot blockedQueue(
            boolean priority,
            GroupPlanner.Item head,
            RouteProjection.AdmissionBlockSemantics semantics) {
        return RouteProjectionTestSupport.queue(
                true, priority, constraints(1, 0L), List.of(head), block(head, semantics));
    }

    private static QueueSnapshot.AdmissionBlock block(
            GroupPlanner.Item head, RouteProjection.AdmissionBlockSemantics semantics) {
        return new QueueSnapshot.AdmissionBlock(head.requestId(), head.enqueueSeq(), semantics);
    }

    private static RouteProjection.AdmissionBlockSemantics semantics(
            RouteProjection.AfterProbeAdmission afterProbe) {
        return new RouteProjection.AdmissionBlockSemantics(
                "HEAD_CAPACITY_BLOCKED", afterProbe, "AFTER_PROBE_CAPACITY_UNKNOWN");
    }
}
