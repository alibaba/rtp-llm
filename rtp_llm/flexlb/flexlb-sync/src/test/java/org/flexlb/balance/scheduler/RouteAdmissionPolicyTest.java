package org.flexlb.balance.scheduler;

import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.QueueView;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.OptionalLong;

import static org.flexlb.balance.scheduler.RouteTtftEstimateTestSupport.ROUTE;
import static org.flexlb.balance.scheduler.RouteTtftEstimateTestSupport.TOKEN_EVALUATOR;
import static org.flexlb.balance.scheduler.RouteTtftEstimateTestSupport.constraints;
import static org.flexlb.balance.scheduler.RouteTtftEstimateTestSupport.estimate;
import static org.flexlb.balance.scheduler.RouteTtftEstimateTestSupport.item;
import static org.flexlb.balance.scheduler.RouteTtftEstimateTestSupport.noCommittedWork;
import static org.flexlb.balance.scheduler.RouteTtftEstimateTestSupport.probe;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Exact observed-head admission policy on top of the pure route timeline. */
class RouteAdmissionPolicyTest {

    @Test
    void prefillSelectionScoresCacheHitsDespiteDecodeDeliveryWait() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueView waiting = blockedQueue(false, head, semantics(
                PrefillTtftEstimate.AfterProbeAdmission.UNAVAILABLE, RoleType.DECODE));
        PrefillTtftEstimate.Probe incoming = probe(99L, 50, 20L, 10L);
        PrefillTtftEstimate.LoadView inputs = new PrefillTtftEstimate.LoadView(waiting, noCommittedWork());

        PrefillTtftEstimate.CandidateView selected = PrefillTtftEstimate.session().estimateViewForSelection(
                inputs, incoming.requestId(), incoming.priority(), incoming.enqueuedAtMs(),
                incoming.expiresAtMs(), incoming.seqLen(), incoming.hitCache(),
                incoming.routingCacheMatchTokens(), TOKEN_EVALUATOR, ROUTE,
                RouteTtftEstimateTestSupport.NOW_MS);

        assertTrue(selected.selectable());
        assertEquals(10L, selected.cacheHitTokens());
        assertEquals(PrefillTtftEstimate.Candidate.State.BLOCKED,
                estimate(waiting, noCommittedWork(), TOKEN_EVALUATOR, incoming, ROUTE).state());
    }

    @Test
    void deliveryOnlyWaitPreservesTheQueueForecast() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        PrefillTtftEstimate.Candidate result = estimate(
                blockedQueue(false, head, null),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertEquals(PrefillTtftEstimate.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(120L), result.estimatedTtftMs());
        assertEquals(PrefillTtftEstimate.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
    }

    @Test
    void fifoObservedHeadBlocksProbeWithoutInventingMilliseconds() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        PrefillTtftEstimate.Candidate result = estimate(
                blockedQueue(false, head, semantics(
                        PrefillTtftEstimate.AfterProbeAdmission.BLOCKED)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 100, 20L, 0L),
                ROUTE);

        assertEquals(PrefillTtftEstimate.Candidate.State.BLOCKED, result.state());
        assertEquals(OptionalLong.empty(), result.estimatedTtftMs());
        assertEquals("HEAD_CAPACITY_BLOCKED", result.detail());
        assertFalse(result.selectable());
    }

    @Test
    void prioritySameOrLowerProbeRemainsBehindObservedHead() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        QueueView blocked = blockedQueue(true, head, semantics(
                PrefillTtftEstimate.AfterProbeAdmission.BLOCKED));

        for (int probePriority : List.of(50, 49)) {
            PrefillTtftEstimate.Candidate result = estimate(
                    blocked, noCommittedWork(), TOKEN_EVALUATOR,
                    probe(99L, probePriority, 20L, 0L),
                    ROUTE);
            assertEquals(PrefillTtftEstimate.Candidate.State.BLOCKED,
                    result.state());
            assertEquals(PrefillTtftEstimate.Candidate.InitialHeadDisposition
                            .BEFORE_PROBE,
                    result.initialHeadDisposition());
        }
    }

    @Test
    void higherPriorityProbeCanRemainHardBlockedByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        PrefillTtftEstimate.Candidate result = estimate(
                blockedQueue(true, head, semantics(
                        PrefillTtftEstimate.AfterProbeAdmission.BLOCKED)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L),
                ROUTE);

        assertEquals(PrefillTtftEstimate.Candidate.State.BLOCKED, result.state());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
    }

    @Test
    void higherPriorityProbeCanBeUnavailableByCapturedSemantics() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        PrefillTtftEstimate.Candidate result = estimate(
                blockedQueue(true, head, semantics(
                        PrefillTtftEstimate.AfterProbeAdmission.UNAVAILABLE,
                        RoleType.DECODE)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L),
                ROUTE);

        assertEquals(PrefillTtftEstimate.Candidate.State.UNAVAILABLE, result.state());
        assertEquals(OptionalLong.empty(), result.estimatedTtftMs());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
        assertEquals(RoleType.DECODE, result.blockerRole());
    }

    @Test
    void decodeCapacityEvidenceSurvivesAnObservedHeadAheadOfProbe() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        PrefillTtftEstimate.Candidate result = estimate(
                blockedQueue(true, head, semantics(
                        PrefillTtftEstimate.AfterProbeAdmission.UNAVAILABLE,
                        RoleType.DECODE)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 40, 20L, 0L),
                ROUTE);

        assertEquals(PrefillTtftEstimate.Candidate.State.BLOCKED, result.state());
        assertEquals(PrefillTtftEstimate.Candidate.InitialHeadDisposition.BEFORE_PROBE,
                result.initialHeadDisposition());
        assertEquals(RoleType.DECODE, result.blockerRole());
    }

    @Test
    void ignoredUnknownWorkPreservesTheAfterProbeCapacityCheck() {
        GroupPlanner.Item head = item(1L, 50, 1L, 100L);
        WorkView unknownWork = RouteTtftEstimateTestSupport.work(
                List.of(), List.of(), 1L);
        PrefillTtftEstimate.Candidate result = estimate(
                blockedQueue(true, head, semantics(
                        PrefillTtftEstimate.AfterProbeAdmission
                                .BLOCKED)),
                unknownWork, TOKEN_EVALUATOR,
                probe(99L, 90, 20L, 0L),
                ROUTE);

        assertEquals(PrefillTtftEstimate.Candidate.State.BLOCKED, result.state());
        assertEquals("AFTER_PROBE_CAPACITY_UNKNOWN", result.detail());
    }

    @Test
    void terminallyPrunedObservedHeadDoesNotBlockProbe() {
        GroupPlanner.Item expired = item(
                1L, 50, 1L, 100L,
                RouteTtftEstimateTestSupport.NOW_MS);
        PrefillTtftEstimate.Candidate result = estimate(
                blockedQueue(false, expired, semantics(
                        PrefillTtftEstimate.AfterProbeAdmission.BLOCKED)),
                noCommittedWork(), TOKEN_EVALUATOR,
                probe(99L, 50, 20L, 0L),
                ROUTE);

        assertEquals(PrefillTtftEstimate.Candidate.State.MODELED, result.state());
        assertEquals(OptionalLong.of(20L), result.estimatedTtftMs());
        assertEquals(PrefillTtftEstimate.Candidate.InitialHeadDisposition
                        .TERMINAL_PRUNED,
                result.initialHeadDisposition());
    }

    private static QueueView blockedQueue(
            boolean priority,
            GroupPlanner.Item head,
            PrefillTtftEstimate.AdmissionBlockSemantics semantics) {
        return RouteTtftEstimateTestSupport.queue(
                priority,
                constraints(1, 0L),
                List.of(head),
                block(head, semantics));
    }

    private static QueueView.AdmissionBlock block(
            GroupPlanner.Item head,
            PrefillTtftEstimate.AdmissionBlockSemantics semantics) {
        return new QueueView.AdmissionBlock(
                head.requestId(), head.enqueueSeq(), semantics);
    }

    private static PrefillTtftEstimate.AdmissionBlockSemantics semantics(
            PrefillTtftEstimate.AfterProbeAdmission afterProbe) {
        return semantics(afterProbe, RoleType.PREFILL);
    }

    private static PrefillTtftEstimate.AdmissionBlockSemantics semantics(
            PrefillTtftEstimate.AfterProbeAdmission afterProbe,
            RoleType blockerRole) {
        return new PrefillTtftEstimate.AdmissionBlockSemantics(
                "HEAD_CAPACITY_BLOCKED",
                afterProbe,
                "AFTER_PROBE_CAPACITY_UNKNOWN",
                blockerRole);
    }
}
