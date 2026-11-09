package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

/**
 * Checks Encoder load selection across health, cache estimates and scheduling modes.
 */
class EncoderStrategyTest {
    private final EncoderSelectionStrategy strategy = new EncoderStrategy();

    @Test
    void selectsLeastConcurrentWorkerWithoutUsingAvailableKvCache() {
        var busy = candidate("busy", true, 4, 0);
        var winner = candidate("zero-kv", true, 2, 100);
        var sameCount = candidate("small-kv", true, 2, 0);
        assertSame(
                winner,
                strategy.select(List.of(busy, winner, sameCount), "group", false, 10).value());
    }

    @Test
    void ignoresDeadWorkersAndIncludesPendingLocalRequests() {
        var dead = candidate("dead", false, 0, 0);
        var pending = candidate("pending", true, 3, 0);
        var winner = candidate("winner", true, 1, 0);
        assertSame(
                winner,
                strategy.select(List.of(dead, pending, winner), "group", false, 10).value());
    }

    @Test
    void returnsExactBlockerWhenNoLiveWorkerExists() {
        var result = strategy.select(List.of(candidate("dead", false, 0, 0)), "group", false, 10);
        assertEquals(PlacementResult.Status.BLOCKED, result.status());
        assertEquals(new PlacementKey(RoleType.ENCODER, "group", null), result.blocker());
    }

    @Test
    void selectsZeroKvWorkerWhenItHasLessConcurrentWork() {
        var busy = candidate("positive", true, 1, 0);
        var winner = candidate("zero", true, 0, 0);
        assertSame(winner, strategy.select(List.of(busy, winner), "group", false, 10).value());
    }

    @Test
    void knownCacheEstimateSelectsLessUncachedWorkEvenWhenItHasMoreRequests() {
        var longJob = candidate("long-job", true, 1, 1000);
        var shortJobs = candidate("short-jobs", true, 4, 40);
        assertSame(
                shortJobs, strategy.select(List.of(longJob, shortJobs), "group", true, 10).value());
    }

    @Test
    void queueModeSkipsWorkersAtEncoderInflightLimit() {
        var atCap = candidate("at-cap", true, 2, 0);
        var winner = candidate("winner", true, 1, 0);
        assertSame(winner, strategy.select(List.of(atCap, winner), "group", false, 2).value());
    }

    @Test
    void queueModeBlocksWhenEveryWorkerIsAtEncoderInflightLimit() {
        var result =
                strategy.select(
                        List.of(candidate("running", true, 1, 0), candidate("pending", true, 1, 0)),
                        "group",
                        false,
                        1);
        assertEquals(PlacementResult.Status.BLOCKED, result.status());
    }

    @Test
    void directModeUsesAnUnboundedRequestLimit() {
        var worker = candidate("only-worker", true, 100, 0);
        assertSame(
                worker,
                strategy.select(List.of(worker), "group", false, Integer.MAX_VALUE).value());
    }

    @Test
    void doesNotSelectAnotherGroup() {
        var other = new Candidate("other", "another-group", true, 0, 0);
        assertEquals(
                PlacementResult.Status.BLOCKED,
                strategy.select(List.of(other), "group", false, 10).status());
    }

    private static Candidate candidate(
            String address, boolean available, long requests, long tokens) {
        return new Candidate(address, "group", available, requests, tokens);
    }

    private record Candidate(
            String address, String group, boolean available, long requestCount, long uncachedTokens)
            implements EncoderSelectionStrategy.Candidate {}
}
