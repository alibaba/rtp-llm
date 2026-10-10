package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;

import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.ROUTE;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.TOKEN_EVALUATOR;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.constraints;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.item;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.noCommittedWork;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.probe;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.project;
import static org.flexlb.balance.scheduler.RouteProjectionTestSupport.queue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/** Exact observed-head admission policy on top of the pure route timeline. */
class RouteAdmissionPolicyTest {

    @Test
    void deliveryOnlyWaitPreservesTheQueueForecast() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RouteProjection.Candidate result = project(
                blockedQueue(false, head, null),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(120L, result.projectedTtftMsValue());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
    }

    @Test
    void fifoObservedHeadBlocksProbeWithoutInventingMilliseconds() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RouteProjection.Candidate result = project(
                blockedQueue(false, head, semantics(
                        RouteProjection.AfterProbeAdmission.BLOCKED)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 100, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.BLOCKED, result.state());
        assertEquals(RouteProjection.Candidate.UNKNOWN, result.projectedTtftMsValue());
        assertEquals("HEAD_CAPACITY_BLOCKED", result.detail());
        assertFalse(result.selectable());
    }

    @Test
    void prioritySameOrLowerProbeRemainsBehindObservedHead() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueSnapshot blocked = blockedQueue(true, head, semantics(
                RouteProjection.AfterProbeAdmission.BLOCKED));

        for (int probePriority : List.of(50, 49)) {
            RouteProjection.Candidate result = project(
                    blocked, noCommittedWork(), TOKEN_EVALUATOR,
                    probe(99L, probePriority, 20L, 0L),
                    ROUTE);
            assertEquals(RouteProjection.Candidate.State.BLOCKED,
                    result.state());
            assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                            .BEFORE_PROBE,
                    result.initialHeadDisposition());
        }
    }

    @Test
    void higherPriorityProbeCanRemainHardBlockedByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RouteProjection.Candidate result = project(
                blockedQueue(true, head, semantics(
                        RouteProjection.AfterProbeAdmission.BLOCKED)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.BLOCKED, result.state());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
    }

    @Test
    void higherPriorityProbeCanBeUnavailableByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RouteProjection.Candidate result = project(
                blockedQueue(true, head, semantics(
                        RouteProjection.AfterProbeAdmission.UNAVAILABLE,
                        RoleType.DECODE)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.UNAVAILABLE, result.state());
        assertEquals(RouteProjection.Candidate.UNKNOWN, result.projectedTtftMsValue());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
        assertEquals(RoleType.DECODE, result.blockerRole());
    }

    @Test
    void decodeCapacityEvidenceSurvivesAnObservedHeadAheadOfProbe() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RouteProjection.Candidate result = project(
                blockedQueue(true, head, semantics(
                        RouteProjection.AfterProbeAdmission.UNAVAILABLE,
                        RoleType.DECODE)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 40, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.BLOCKED, result.state());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
        assertEquals(RoleType.DECODE, result.blockerRole());
    }

    @ParameterizedTest
    @EnumSource(RouteProjection.AfterProbeAdmission.class)
    void unknownEngineCursorCannotProveOvertakeOfBlockedHead(RouteProjection.AfterProbeAdmission afterProbe) {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        WorkSnapshot unknownWork = RouteProjectionTestSupport.work(
                List.of(), List.of(), 1L);
        RouteProjection.Candidate result = project(
                blockedQueue(true, head, semantics(afterProbe, RoleType.DECODE)),
                unknownWork, TOKEN_EVALUATOR,
                new RouteProjectionTestSupport.Probe(99L, 90, RouteProjectionTestSupport.NOW_MS,
                        Long.MAX_VALUE, 20L, 7L, 11L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.BLOCKED, result.state());
        assertEquals("HEAD_CAPACITY_BLOCKED", result.detail());
        assertEquals(RouteProjection.Candidate.UNKNOWN, result.projectedTtftMsValue());
        assertEquals(afterProbe == RouteProjection.AfterProbeAdmission.UNAVAILABLE ? RoleType.DECODE : null,
                result.blockerRole());
        assertEquals(15L, result.incomingPrefillMs());
        assertEquals(7L, result.cacheHitTokens());
        assertEquals(11L, result.routingCacheMatchTokens());
    }

    @Test
    void terminallyPrunedObservedHeadDoesNotBlockProbe() {
        GroupPlanner.Item expired = item(
                1L, 50, 1L, 100L,
                RouteProjectionTestSupport.NOW_MS);
        RouteProjection.Candidate result = project(
                blockedQueue(false, expired, semantics(
                        RouteProjection.AfterProbeAdmission.BLOCKED)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition
                        .TERMINAL_PRUNED,
                result.initialHeadDisposition());
    }

    @Test
    void directProjectionIgnoresPriorQueueAdmissionObservation() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueSnapshot queue = RouteProjectionTestSupport.queue(
                false,
                false,
                constraints(1, 0L),
                List.of(head),
                block(head, semantics(
                        RouteProjection.AfterProbeAdmission.BLOCKED)));

        RouteProjection.Candidate result = project(
                queue, noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertEquals(RouteProjection.Candidate.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(RouteProjection.Candidate.InitialHeadDisposition.NONE,
                result.initialHeadDisposition());
    }

    private static QueueSnapshot blockedQueue(
            boolean priority,
            GroupPlanner.Item head,
            RouteProjection.AdmissionBlockSemantics semantics) {
        return RouteProjectionTestSupport.queue(
                true,
                priority,
                constraints(1, 0L),
                List.of(head),
                block(head, semantics));
    }

    private static QueueSnapshot.AdmissionBlock block(
            GroupPlanner.Item head,
            RouteProjection.AdmissionBlockSemantics semantics) {
        return new QueueSnapshot.AdmissionBlock(
                head.requestId(), head.enqueueSeq(), semantics);
    }

    private static RouteProjection.AdmissionBlockSemantics semantics(
            RouteProjection.AfterProbeAdmission afterProbe) {
        return semantics(afterProbe, RoleType.PREFILL);
    }

    private static RouteProjection.AdmissionBlockSemantics semantics(
            RouteProjection.AfterProbeAdmission afterProbe,
            RoleType blockerRole) {
        return new RouteProjection.AdmissionBlockSemantics(
                "HEAD_CAPACITY_BLOCKED",
                afterProbe,
                "AFTER_PROBE_CAPACITY_UNKNOWN",
                blockerRole);
    }
}
