package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher.PreparedSubmission;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.util.Failures;

import java.util.List;
import java.util.OptionalLong;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

/**
 * Mode-specific part of delivery. Grouping chooses an ordered candidate group;
 * WorkerBatcher commits the prepared transaction, then invokes this strategy
 * outside the queue lock to submit a batch or publish individual routes.
 */
public interface DeliveryStrategy {

    /** Reserve the largest feasible prefix without mutating queue ownership. */
    DeliveryTransaction prepare(
            List<RequestRoute> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs);

    /** Prepare one ordered resource prefix; prediction and acquisition share exact request ownership. */
    static DeliveryTransaction prepareMembers(List<RequestRoute> candidates, PrefillTimePredictor.Evaluator evaluator,
            Supplier<CapacityBoundary.Attempt<PreparedSubmission>> prepareSubmission, LongSupplier batchIds) {
        var transaction = new DeliveryTransaction(candidates, prepareSubmission != null);
        String failureMessage = prepareSubmission == null ? "route delivery failed" : "batch delivery failed";
        Throwable failure = null;
        boolean prepared = false;
        try {
            for (RequestRoute item : candidates) {
                CapacityBoundary boundary;
                RequestContext context = item.ctx();
                synchronized (context) {
                    boundary = context.scheduler().ownsPreparedDeliveryLocked(context, item)
                            ? transaction.append(item, transaction.batch == null
                                    ? PrefillPredictionBoundary.predictSingleRequestMs(evaluator, item.seqLen(), item.hitCache()) : 0L,
                                    prepareSubmission, batchIds) : CapacityBoundary.OWNERSHIP_LOST;
                }
                if (boundary != null) {
                    transaction.blockedItem = item;
                    transaction.blockedResult = boundary;
                    break;
                }
            }
            transaction.finishPreparation();
            prepared = !transaction.members.isEmpty();
            return transaction;
        } catch (Throwable preparationFailure) {
            failure = preparationFailure;
            throw Failures.propagate(failure, failureMessage);
        } finally {
            if (!prepared) {
                Throwable cleanup = Failures.close(transaction);
                if (failure != null) {
                    Failures.append(failure, cleanup);
                } else if (transaction.batch == null) {
                    Failures.rethrow(cleanup, failureMessage);
                } else if (cleanup != null) {
                    if (transaction.blockedResult != null
                            && transaction.blockedResult.status() == CapacityBoundary.Status.FAILED
                            && transaction.blockedResult.cause() != cleanup) {
                        cleanup.addSuppressed(transaction.blockedResult.cause());
                    }
                    transaction.blockedResult = CapacityBoundary.failed(cleanup);
                }
            }
        }
    }

    /** Fresh callback for GroupPlanner's strictly growing prefixes in one select call. */
    GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
            PrefillTimePredictor.Evaluator evaluator);

    /** Pure projection behavior paired with this live delivery strategy. */
    RouteProjection.DeliveryProjection projectionPolicy();

    /** Hand committed resources to the transport or publish individual routes outside the queue lock. */
    void deliver(DeliveryTransaction transaction, String decisionReason,
                 int remainingQueueDepth, WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator);

    /** Fail each unsent member, then release its temporary delivery resources even if notification fails. */
    static void failUnsentDelivery(DeliveryTransaction transaction, Throwable cause, boolean executorOwner) {
        if (transaction.batchDelivery()) {
            synchronized (transaction) {
                if (transaction.tryReclaimUnsent(executorOwner)) {
                    notifyUnsentMembers(transaction, cause != null ? cause
                            : new IllegalStateException("delivery returned without resolving owner"));
                }
            }
        } else if (transaction.tryReclaimUnsent(executorOwner)) {
            notifyUnsentMembers(transaction, cause);
        }
    }

    private static void notifyUnsentMembers(DeliveryTransaction transaction, Throwable cause) {
        Throwable failure = Failures.run(null, transaction::closeSubmission);
        try {
            for (RequestRoute item : transaction.items()) {
                failure = Failures.run(failure, () -> item.ctx().scheduler().failDeliveryPreparation(item, cause));
            }
        } finally {
            failure = Failures.append(failure, transaction.finishDelivery());
        }
        Failures.rethrow(failure, "unsent delivery cleanup failed");
    }
}
