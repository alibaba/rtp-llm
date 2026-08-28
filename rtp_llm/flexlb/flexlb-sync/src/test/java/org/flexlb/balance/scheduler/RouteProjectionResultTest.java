package org.flexlb.balance.scheduler;

import org.flexlb.balance.projection.RouteProjection;
import org.junit.jupiter.api.Test;

import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertThrows;

class RouteProjectionResultTest {

    @Test
    void rejectsNegativeModeledComponentsInsteadOfSilentlyClampingThem() {
        assertThrows(IllegalArgumentException.class,
                () -> modeled(-1L));
        assertThrows(IllegalArgumentException.class,
                () -> new RouteProjection.Result(
                        RouteProjection.Result.State.MODELED,
                        OptionalLong.of(-1L),
                        OptionalLong.of(0L),
                        0L,
                        RouteProjection.Result.InitialHeadDisposition.NONE,
                        "test"));
        assertThrows(IllegalArgumentException.class,
                () -> new RouteProjection.Result(
                        RouteProjection.Result.State.MODELED,
                        OptionalLong.of(0L),
                        OptionalLong.of(-1L),
                        0L,
                        RouteProjection.Result.InitialHeadDisposition.NONE,
                        "test"));
    }

    @Test
    void onlyModeledResultsMayCarryTtftOrDrain() {
        assertThrows(IllegalArgumentException.class,
                () -> new RouteProjection.Result(
                        RouteProjection.Result.State.UNAVAILABLE,
                        OptionalLong.of(1L), OptionalLong.empty(), 0L,
                        RouteProjection.Result.InitialHeadDisposition.NONE,
                        "test"));
        assertThrows(IllegalArgumentException.class,
                () -> new RouteProjection.Result(
                        RouteProjection.Result.State.BLOCKED,
                        OptionalLong.empty(), OptionalLong.of(1L), 0L,
                        RouteProjection.Result.InitialHeadDisposition.NONE,
                        "test"));
    }

    private static RouteProjection.Result modeled(long incomingPrefillMs) {
        return new RouteProjection.Result(
                RouteProjection.Result.State.MODELED,
                OptionalLong.of(1L),
                OptionalLong.of(1L),
                incomingPrefillMs,
                RouteProjection.Result.InitialHeadDisposition.NONE,
                "test");
    }
}
