package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.WorkSnapshot;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.List;

import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.ROUTE;
import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.TOKEN_EVALUATOR;
import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.constraints;
import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.item;
import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.noCommittedWork;
import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.probe;
import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.project;
import static org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.queue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/** Exact observed-head admission policy on top of the pure route timeline. */
class RouteAdmissionPolicyTest {

    @Test
    void deliveryOnlyWaitPreservesTheQueueForecast() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RequestTimeEstimation.Candidate result =
                project(
                        blockedQueue(false, head, null),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 50, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.MODELED, result.state());
        assertEquals(120L, result.projectedTtftMsValue());
        assertEquals(
                RequestTimeEstimation.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
    }

    @Test
    void fifoObservedHeadBlocksProbeWithoutInventingMilliseconds() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RequestTimeEstimation.Candidate result =
                project(
                        blockedQueue(
                                false,
                                head,
                                semantics(RequestTimeEstimation.AfterProbeAdmission.BLOCKED)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 100, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.BLOCKED, result.state());
        assertEquals(RequestTimeEstimation.Candidate.UNKNOWN, result.projectedTtftMsValue());
        assertEquals("HEAD_CAPACITY_BLOCKED", result.detail());
        assertFalse(result.selectable());
    }

    @Test
    void prioritySameOrLowerProbeRemainsBehindObservedHead() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueSnapshot blocked =
                blockedQueue(
                        true, head, semantics(RequestTimeEstimation.AfterProbeAdmission.BLOCKED));

        for (int probePriority : List.of(50, 49)) {
            RequestTimeEstimation.Candidate result =
                    project(
                            blocked,
                            noCommittedWork(),
                            TOKEN_EVALUATOR,
                            probe(99L, probePriority, 20L, 0L),
                            ROUTE);
            assertEquals(RequestTimeEstimation.Candidate.State.BLOCKED, result.state());
            assertEquals(
                    RequestTimeEstimation.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                    result.initialHeadDisposition());
        }
    }

    @Test
    void higherPriorityProbeCanRemainHardBlockedByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RequestTimeEstimation.Candidate result =
                project(
                        blockedQueue(
                                true,
                                head,
                                semantics(RequestTimeEstimation.AfterProbeAdmission.BLOCKED)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 90, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.BLOCKED, result.state());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
    }

    @Test
    void higherPriorityProbeCanBeUnavailableByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RequestTimeEstimation.Candidate result =
                project(
                        blockedQueue(
                                true,
                                head,
                                semantics(RequestTimeEstimation.AfterProbeAdmission.UNAVAILABLE)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 90, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.UNAVAILABLE, result.state());
        assertEquals(RequestTimeEstimation.Candidate.UNKNOWN, result.projectedTtftMsValue());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
    }

    @Test
    void decodeCapacityEvidenceSurvivesAnObservedHeadAheadOfProbe() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        RequestTimeEstimation.Candidate result =
                project(
                        blockedQueue(
                                true,
                                head,
                                semantics(RequestTimeEstimation.AfterProbeAdmission.UNAVAILABLE)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 40, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.BLOCKED, result.state());
        assertEquals(
                RequestTimeEstimation.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
    }

    @ParameterizedTest
    @EnumSource(RequestTimeEstimation.AfterProbeAdmission.class)
    void unknownEngineCursorCannotProveOvertakeOfBlockedHead(
            RequestTimeEstimation.AfterProbeAdmission afterProbe) {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        WorkSnapshot unknownWork = RequestTimeEstimationTestSupport.work(List.of(), List.of(), 1L);
        RequestTimeEstimation.Candidate result =
                project(
                        blockedQueue(true, head, semantics(afterProbe)),
                        unknownWork,
                        TOKEN_EVALUATOR,
                        new RequestTimeEstimationTestSupport.Probe(
                                99L,
                                90,
                                RequestTimeEstimationTestSupport.NOW_MS,
                                Long.MAX_VALUE,
                                20L,
                                7L,
                                11L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.BLOCKED, result.state());
        assertEquals("HEAD_CAPACITY_BLOCKED", result.detail());
        assertEquals(RequestTimeEstimation.Candidate.UNKNOWN, result.projectedTtftMsValue());
        assertEquals(15L, result.incomingPrefillMs());
        assertEquals(7L, result.cacheHitTokens());
        assertEquals(11L, result.routingCacheMatchTokens());
    }

    @Test
    void terminallyPrunedObservedHeadDoesNotBlockProbe() {
        GroupPlanner.Item expired = item(1L, 50, 1L, 100L, RequestTimeEstimationTestSupport.NOW_MS);
        RequestTimeEstimation.Candidate result =
                project(
                        blockedQueue(
                                false,
                                expired,
                                semantics(RequestTimeEstimation.AfterProbeAdmission.BLOCKED)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 50, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(
                RequestTimeEstimation.Candidate.InitialHeadDisposition.TERMINAL_PRUNED,
                result.initialHeadDisposition());
    }

    @Test
    void directProjectionIgnoresPriorQueueAdmissionObservation() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueSnapshot queue =
                RequestTimeEstimationTestSupport.queue(
                        false,
                        false,
                        constraints(1, 0L),
                        List.of(head),
                        block(head, semantics(RequestTimeEstimation.AfterProbeAdmission.BLOCKED)));

        RequestTimeEstimation.Candidate result =
                project(queue, noCommittedWork(), TOKEN_EVALUATOR, probe(99L, 50, 20L, 0L), ROUTE);

        assertEquals(RequestTimeEstimation.Candidate.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(
                RequestTimeEstimation.Candidate.InitialHeadDisposition.NONE,
                result.initialHeadDisposition());
    }

    private static QueueSnapshot blockedQueue(
            boolean priority,
            GroupPlanner.Item head,
            RequestTimeEstimation.AdmissionBlockSemantics semantics) {
        return RequestTimeEstimationTestSupport.queue(
                true, priority, constraints(1, 0L), List.of(head), block(head, semantics));
    }

    private static QueueSnapshot.AdmissionBlock block(
            GroupPlanner.Item head, RequestTimeEstimation.AdmissionBlockSemantics semantics) {
        return new QueueSnapshot.AdmissionBlock(head.requestId(), head.enqueueSeq(), semantics);
    }

    private static RequestTimeEstimation.AdmissionBlockSemantics semantics(
            RequestTimeEstimation.AfterProbeAdmission afterProbe) {
        return new RequestTimeEstimation.AdmissionBlockSemantics(
                "HEAD_CAPACITY_BLOCKED", afterProbe, "AFTER_PROBE_CAPACITY_UNKNOWN");
    }
}
