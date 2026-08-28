package org.flexlb.balance.projection;

import java.util.OptionalLong;

/**
 * Applies one observed worker admission wait after pure service-time
 * projection. It never invents a capacity release duration.
 */
final class RouteAdmissionPolicy {

    private RouteAdmissionPolicy() {
    }

    static RouteProjection.Result apply(
            QueueSnapshot queue,
            RouteProjection.Result timeline) {
        QueueSnapshot.AdmissionBlock observation = queue.admissionBlock();
        if (!queue.queueScheduling()
                || observation == null
                || (timeline.state() != RouteProjection.Result.State.MODELED
                        && !timeline.engineWorkUnmodeled())) {
            return timeline;
        }
        if (timeline.engineWorkUnmodeled()) {
            // Without an Engine cursor there is no honest way to prove that
            // the probe overtakes this exact blocked head. Keep the hard block
            // and, critically, do not manufacture a MODELED TTFT.
            return blocked(timeline, observation.semantics().blockedDetail());
        }

        return switch (timeline.initialHeadDisposition()) {
            case TERMINAL_PRUNED -> timeline;
            case BEFORE_PROBE -> blocked(
                    timeline, observation.semantics().blockedDetail());
            case NONE -> throw new IllegalStateException(
                    "admission-blocked ACTIVE head was not projected");
            case AFTER_PROBE -> applyAfterProbe(
                    observation.semantics(), timeline);
        };
    }

    private static RouteProjection.Result applyAfterProbe(
            RouteProjection.AdmissionBlockSemantics semantics,
            RouteProjection.Result timeline) {
        return switch (semantics.afterProbe()) {
            case BLOCKED -> blocked(timeline, semantics.afterProbeDetail());
            case TTFT_KNOWN_DRAIN_UNKNOWN -> knownTtftUnknownDrain(
                    timeline, semantics.afterProbeDetail());
            case UNAVAILABLE -> unavailable(
                    timeline, semantics.afterProbeDetail());
        };
    }

    private static RouteProjection.Result blocked(
            RouteProjection.Result timeline,
            String detail) {
        return copy(
                timeline,
                RouteProjection.Result.State.BLOCKED,
                OptionalLong.empty(),
                OptionalLong.empty(),
                detail);
    }

    private static RouteProjection.Result knownTtftUnknownDrain(
            RouteProjection.Result timeline,
            String detail) {
        return copy(
                timeline,
                RouteProjection.Result.State.MODELED,
                timeline.projectedTtftMs(),
                OptionalLong.empty(),
                detail);
    }

    private static RouteProjection.Result unavailable(
            RouteProjection.Result timeline, String detail) {
        return copy(
                timeline,
                RouteProjection.Result.State.UNAVAILABLE,
                OptionalLong.empty(),
                OptionalLong.empty(),
                detail);
    }

    private static RouteProjection.Result copy(
            RouteProjection.Result source,
            RouteProjection.Result.State state,
            OptionalLong projectedTtftMs,
            OptionalLong projectedDrainMs,
            String detail) {
        return new RouteProjection.Result(
                state,
                projectedTtftMs,
                projectedDrainMs,
                source.incomingPrefillMs(),
                source.initialHeadDisposition(),
                detail);
    }
}
