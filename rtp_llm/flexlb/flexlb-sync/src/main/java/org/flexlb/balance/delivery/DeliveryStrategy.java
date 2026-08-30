package org.flexlb.balance.delivery;

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
    PreparedDelivery prepare(
            List<DeliveryItem> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs);

    /**
     * Pure planning duration for an exact group. The return value keeps
     * fractional milliseconds so GroupPlanner can compare exact boundaries.
     */
    double projectGroupDurationMs(
            List<DeliveryItem> items,
            PrefillTimePredictor.Evaluator evaluator);

    /** Pure projection behavior paired with this live delivery strategy. */
    RouteProjection.DeliveryProjection projectionPolicy();

    /**
     * One invocation's pre-commit transaction. Closing releases every resource
     * that has not crossed the queue commit boundary.
     */
    interface PreparedDelivery extends AutoCloseable {

        List<DeliveryItem> items();

        /** First candidate not covered by the prepared prefix, if any. */
        SelectionBoundary boundary();

        /** Move all prepared resources to one post-commit owner. */
        Handoff commitOwnershipUnderLock();

        @Override
        void close();
    }

    /** Sole owner after the canonical queue commit. */
    interface Handoff {

        List<DeliveryItem> items();

        void deliver(DeliveryMetadata metadata);

        void failBeforeDelivery(Throwable cause);
    }

    /** First candidate not covered by the prepared prefix. */
    record SelectionBoundary(
            DeliveryItem item,
            CapacityBoundary result) {
    }
}
