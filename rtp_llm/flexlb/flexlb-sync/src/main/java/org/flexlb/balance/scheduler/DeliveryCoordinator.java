package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.ScheduledRequest;
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

    private final String workerKey;
    private final DeliveryStrategy mode;

    DeliveryCoordinator(String workerKey, DeliveryStrategy mode) {
        this.workerKey = Objects.requireNonNull(workerKey, "workerKey");
        this.mode = Objects.requireNonNull(mode, "mode");
    }

    BatcherCycleResult deliver(
            BatcherContext queue,
            List<ScheduledRequest> candidates,
            DeliveryMetadata proposedMetadata,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        if (candidates.isEmpty() || !queue.selectionStillOwned(candidates)) {
            return BatcherCycleResult.NO_ACTION;
        }

        try (DeliveryStrategy.Transaction transaction = mode.prepare(
                deliveryItems(candidates), evaluator, plannedPredictionMs)) {
            if (transaction.items().isEmpty()) {
                return queue.commitBoundary(
                        transaction.blockedItem(),
                        transaction.blockedResult());
            }

            BatcherCycleResult admitted =
                    queue.commitPreparedSelection(
                            transaction,
                            proposedMetadata.decisionReason());
            if (admitted == null) {
                return BatcherCycleResult.NO_ACTION;
            }

            // Only the lock-linearized result owns handoff metadata. In
            // particular, a capacity prefix changes both reason and depth.
            handoff(transaction, admitted.metadata());
            return admitted;
        }
    }

    double projectGroupDurationMs(
            List<ScheduledRequest> items,
            PrefillTimePredictor.Evaluator evaluator) {
        return mode.projectGroupDurationMs(deliveryItems(items), evaluator);
    }

    RouteProjection.DeliveryProjection projectionPolicy() {
        return mode.projectionPolicy();
    }

    private void handoff(
            DeliveryStrategy.Transaction transaction,
            DeliveryMetadata metadata) {
        Throwable deliveryFailure = null;
        try {
            transaction.handoff(metadata);
        } catch (Throwable failure) {
            deliveryFailure = failure;
        }
        Throwable unresolved = deliveryFailure != null
                ? deliveryFailure
                : new IllegalStateException(
                        "delivery returned without resolving owner");
        try {
            transaction.abort(unresolved);
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

    /** ScheduledRequest is the sole production ScheduledRequest implementation here. */
    @SuppressWarnings("unchecked")
    private static List<ScheduledRequest> deliveryItems(List<ScheduledRequest> items) {
        return (List<ScheduledRequest>) (List<?>) items;
    }
}
