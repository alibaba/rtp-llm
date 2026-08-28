package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;

import java.util.List;
import java.util.OptionalLong;

/**
 * Mode-specific part of delivery. Grouping chooses an ordered candidate group;
 * the scheduler coordinates prepare, queue commit, and handoff while this mode
 * owns only its exact capacity and transport resources.
 */
public interface DeliveryStrategy {

    /** Reserve the largest feasible prefix without mutating queue ownership. */
    Transaction prepare(
            List<ScheduledRequest> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs);

    /**
     * Pure planning duration for an exact group. The return value keeps
     * fractional milliseconds so GroupPlanner can compare exact boundaries.
     */
    double projectGroupDurationMs(
            List<ScheduledRequest> items,
            PrefillTimePredictor.Evaluator evaluator);

    /** Pure projection behavior paired with this live delivery strategy. */
    RouteProjection.DeliveryProjection projectionPolicy();

    /** One delivery transaction across prepare, queue commit, and handoff. */
    interface Transaction extends AutoCloseable {

        List<ScheduledRequest> items();

        /** First candidate not covered by this transaction, if any. */
        ScheduledRequest blockedItem();

        CapacityBoundary blockedResult();

        /** Cross the canonical queue commit while its lock is held. */
        void commitUnderLock();

        /** Transfer committed ownership to the configured delivery mode. */
        void handoff(String decisionReason, int remainingQueueDepth);

        /** Resolve any committed ownership that handoff did not transfer. */
        void abort(Throwable cause);

        @Override
        void close();
    }
}
