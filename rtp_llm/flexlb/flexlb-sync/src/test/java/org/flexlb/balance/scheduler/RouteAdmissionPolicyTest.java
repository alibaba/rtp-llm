package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RequestTimeEstimation.ProjectionQueueSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestTimeEstimationTestSupport.Candidate;
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

/**
 * Exact observed-head admission policy on top of the pure route timeline.
 */
class RouteAdmissionPolicyTest {

    @Test
    void oneProjectionPreservesRankingAndRefreshesAdmissionForEachWorker() {
        var head = item(1L, 50, 1L, 100L);
        var estimator = RequestTimeEstimation.current();
        for (boolean blocked : List.of(true, false, true)) {
            var queue =
                    blockedQueue(
                            false,
                            head,
                            blocked ? semantics(RequestTimeEstimation.State.BLOCKED) : null);
            var inputs =
                    new RequestTimeEstimation.Inputs(
                            queue, noCommittedWork(), 0L, TOKEN_EVALUATOR, ROUTE);
            var result =
                    estimator.projectCandidate(
                            inputs,
                            "99",
                            50,
                            RequestTimeEstimationTestSupport.NOW_MS,
                            Long.MAX_VALUE,
                            20L,
                            0L,
                            0L,
                            RequestTimeEstimationTestSupport.NOW_MS);

            assertEquals(RequestTimeEstimation.State.MODELED, result.state());
            assertEquals(120L, result.projectedTtftMsValue());
            assertEquals(
                    blocked
                            ? RequestTimeEstimation.State.BLOCKED
                            : RequestTimeEstimation.State.MODELED,
                    result.admissionState());
            assertEquals(
                    blocked ? "HEAD_CAPACITY_BLOCKED" : result.detail(), result.admissionDetail());
        }
    }

    @Test
    void deliveryOnlyWaitPreservesTheQueueForecast() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        Candidate result =
                project(
                        blockedQueue(false, head, null),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 50, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.State.MODELED, result.state());
        assertEquals(120L, result.projectedTtftMsValue());
        assertEquals(
                RequestTimeEstimation.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
    }

    @Test
    void fifoObservedHeadBlocksProbeWithoutInventingMilliseconds() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        Candidate result =
                project(
                        blockedQueue(false, head, semantics(RequestTimeEstimation.State.BLOCKED)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 100, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.State.BLOCKED, result.state());
        assertEquals(RequestTimeEstimation.UNKNOWN, result.projectedTtftMsValue());
        assertEquals("HEAD_CAPACITY_BLOCKED", result.detail());
        assertFalse(result.selectable());
    }

    @Test
    void prioritySameOrLowerProbeRemainsBehindObservedHead() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        ProjectionQueueSnapshot blocked =
                blockedQueue(true, head, semantics(RequestTimeEstimation.State.BLOCKED));

        for (int probePriority : List.of(50, 49)) {
            Candidate result =
                    project(
                            blocked,
                            noCommittedWork(),
                            TOKEN_EVALUATOR,
                            probe(99L, probePriority, 20L, 0L),
                            ROUTE);
            assertEquals(RequestTimeEstimation.State.BLOCKED, result.state());
            assertEquals(
                    RequestTimeEstimation.InitialHeadDisposition.BEFORE_PROBE,
                    result.initialHeadDisposition());
        }
    }

    @Test
    void higherPriorityProbeCanRemainHardBlockedByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        Candidate result =
                project(
                        blockedQueue(true, head, semantics(RequestTimeEstimation.State.BLOCKED)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 90, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.State.BLOCKED, result.state());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
    }

    @Test
    void higherPriorityProbeCanBeUnavailableByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        Candidate result =
                project(
                        blockedQueue(
                                true, head, semantics(RequestTimeEstimation.State.UNAVAILABLE)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 90, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.State.UNAVAILABLE, result.state());
        assertEquals(RequestTimeEstimation.UNKNOWN, result.projectedTtftMsValue());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
    }

    @Test
    void decodeCapacityEvidenceSurvivesAnObservedHeadAheadOfProbe() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        Candidate result =
                project(
                        blockedQueue(
                                true, head, semantics(RequestTimeEstimation.State.UNAVAILABLE)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 40, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.State.BLOCKED, result.state());
        assertEquals(
                RequestTimeEstimation.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
    }

    @ParameterizedTest
    @EnumSource(
            value = RequestTimeEstimation.State.class,
            names = {"BLOCKED", "UNAVAILABLE"})
    void unknownWorkPreservesCapturedAdmissionSemanticsForHigherPriorityProbe(
            RequestTimeEstimation.State afterProbe) {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        WorkSnapshot unknownWork = RequestTimeEstimationTestSupport.work(List.of(), List.of(), 1L);
        Candidate result =
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

        assertEquals(
                afterProbe == RequestTimeEstimation.State.BLOCKED
                        ? RequestTimeEstimation.State.BLOCKED
                        : RequestTimeEstimation.State.UNAVAILABLE,
                result.state());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
        assertEquals(RequestTimeEstimation.UNKNOWN, result.projectedTtftMsValue());
        assertEquals(15L, result.incomingPrefillMs());
        assertEquals(7L, result.cacheHitTokens());
        assertEquals(11L, result.routingCacheMatchTokens());
    }

    @Test
    void terminallyPrunedObservedHeadDoesNotBlockProbe() {
        GroupPlanner.Item expired = item(1L, 50, 1L, 100L, RequestTimeEstimationTestSupport.NOW_MS);
        Candidate result =
                project(
                        blockedQueue(
                                false, expired, semantics(RequestTimeEstimation.State.BLOCKED)),
                        noCommittedWork(),
                        TOKEN_EVALUATOR,
                        probe(99L, 50, 20L, 0L),
                        ROUTE);

        assertEquals(RequestTimeEstimation.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(
                RequestTimeEstimation.InitialHeadDisposition.TERMINAL_PRUNED,
                result.initialHeadDisposition());
    }

    @Test
    void directProjectionIgnoresPriorQueueAdmissionObservation() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        ProjectionQueueSnapshot queue =
                RequestTimeEstimationTestSupport.queue(
                        false,
                        false,
                        constraints(1, 0L),
                        List.of(head),
                        block(head, semantics(RequestTimeEstimation.State.BLOCKED)));

        Candidate result =
                project(queue, noCommittedWork(), TOKEN_EVALUATOR, probe(99L, 50, 20L, 0L), ROUTE);

        assertEquals(RequestTimeEstimation.State.MODELED, result.state());
        assertEquals(20L, result.projectedTtftMsValue());
        assertEquals(
                RequestTimeEstimation.InitialHeadDisposition.NONE, result.initialHeadDisposition());
    }

    private static ProjectionQueueSnapshot blockedQueue(
            boolean priority,
            GroupPlanner.Item head,
            RequestTimeEstimation.AdmissionBlockSemantics semantics) {
        return RequestTimeEstimationTestSupport.queue(
                true, priority, constraints(1, 0L), List.of(head), block(head, semantics));
    }

    private static ProjectionQueueSnapshot.AdmissionBlock block(
            GroupPlanner.Item head, RequestTimeEstimation.AdmissionBlockSemantics semantics) {
        return new ProjectionQueueSnapshot.AdmissionBlock(
                head.requestId(), head.enqueueSeq(), semantics);
    }

    private static RequestTimeEstimation.AdmissionBlockSemantics semantics(
            RequestTimeEstimation.State afterProbe) {
        return new RequestTimeEstimation.AdmissionBlockSemantics(
                "HEAD_CAPACITY_BLOCKED", afterProbe, "AFTER_PROBE_CAPACITY_UNKNOWN");
    }
}
