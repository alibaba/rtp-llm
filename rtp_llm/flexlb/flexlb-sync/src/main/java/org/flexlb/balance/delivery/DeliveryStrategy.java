package org.flexlb.balance.delivery;

import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;

import java.util.List;
import java.util.OptionalLong;

/**
 * One immutable delivery policy selected when a worker batcher is created.
 * Grouping decides which requests are ready; this strategy owns how that exact
 * group reserves capacity, crosses the queue ownership boundary, and is
 * published to its transport.
 */
public interface DeliveryStrategy {

    /** Reserve and deliver the largest feasible prefix of one ordered group. */
    <R> R admitAndDeliver(
            List<DeliveryItem> candidates,
            DeliveryMetadata metadata,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPrediction,
            DeliveryContext<R> context);

    /**
     * Pure planning duration for an exact group. The return value keeps
     * fractional milliseconds so GroupPlanner can compare exact boundaries.
     */
    double projectGroupDurationMs(
            List<DeliveryItem> items,
            PrefillTimePredictor.Evaluator evaluator);

    /** Pure projection behavior paired with this live delivery strategy. */
    RouteProjection.DeliveryProjection projectionPolicy();
}
