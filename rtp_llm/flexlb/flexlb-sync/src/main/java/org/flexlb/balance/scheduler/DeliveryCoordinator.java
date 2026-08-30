package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.DeliveryMetadata;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/**
 * Visible delivery transaction: prepare outside the queue lock, commit the
 * exact prefix under that lock, then hand its resources to the selected mode.
 */
final class DeliveryCoordinator {

    record CommittedSelection(
            DeliveryStrategy.Handoff handoff,
            BatcherCycleResult.Admitted result) {

        CommittedSelection {
            Objects.requireNonNull(handoff, "handoff");
            Objects.requireNonNull(result, "result");
        }
    }

    private final String workerKey;
    private final DeliveryStrategy mode;

    DeliveryCoordinator(String workerKey, DeliveryStrategy mode) {
        this.workerKey = Objects.requireNonNull(workerKey, "workerKey");
        this.mode = Objects.requireNonNull(mode, "mode");
    }

    BatcherCycleResult deliver(
            BatcherContext queue,
            List<BatchItem> candidates,
            DeliveryMetadata proposedMetadata,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        if (candidates.isEmpty() || !queue.selectionStillOwned(candidates)) {
            return BatcherCycleResult.Outcome.NO_ACTION;
        }

        try (DeliveryStrategy.PreparedDelivery prepared = mode.prepare(
                deliveryItems(candidates), evaluator, plannedPredictionMs)) {
            if (prepared.items().isEmpty()) {
                return queue.commitBoundary(prepared.boundary());
            }

            CommittedSelection committed = queue.commitPreparedSelection(
                    prepared, proposedMetadata.decisionReason());
            if (committed == null) {
                return BatcherCycleResult.Outcome.NO_ACTION;
            }

            // Only the lock-linearized result owns handoff metadata. In
            // particular, a capacity prefix changes both reason and depth.
            handoff(committed.handoff(), committed.result().metadata());
            return committed.result();
        }
    }

    double projectGroupDurationMs(
            List<BatchItem> items,
            PrefillTimePredictor.Evaluator evaluator) {
        return mode.projectGroupDurationMs(deliveryItems(items), evaluator);
    }

    RouteProjection.DeliveryProjection projectionPolicy() {
        return mode.projectionPolicy();
    }

    private void handoff(
            DeliveryStrategy.Handoff handoff,
            DeliveryMetadata metadata) {
        Throwable deliveryFailure = null;
        try {
            handoff.deliver(metadata);
        } catch (Throwable failure) {
            deliveryFailure = failure;
        }
        Throwable unresolved = deliveryFailure != null
                ? deliveryFailure
                : new IllegalStateException(
                        "delivery returned without resolving owner");
        try {
            handoff.failBeforeDelivery(unresolved);
        } catch (Throwable cleanupFailure) {
            if (deliveryFailure == null) {
                deliveryFailure = cleanupFailure;
            } else if (deliveryFailure != cleanupFailure) {
                deliveryFailure.addSuppressed(cleanupFailure);
            }
        }
        if (deliveryFailure != null) {
            Logger.error("WorkerBatcher[{}] committed delivery failed",
                    workerKey, deliveryFailure);
        }
    }

    /** BatchItem is the sole production DeliveryItem implementation here. */
    @SuppressWarnings("unchecked")
    private static List<DeliveryItem> deliveryItems(List<BatchItem> items) {
        return (List<DeliveryItem>) (List<?>) items;
    }
}
