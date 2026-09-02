package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;
import java.util.function.BiConsumer;

/** Transport admission and submission required by batch delivery. */
public interface BatchSubmissionPort {

    CapacityBoundary.Attempt<PreparedSubmission> tryPrepareSubmission();

    /**
     * One prepared dispatcher admission.
     *
     * <p>A successful {@code submitBatch} moves ownership
     * to the dispatcher. The subsequent try-with-resources {@link #close()}
     * must then be a no-op. A repeated or otherwise illegal submit must throw.
     * Recoverable cleanup failures must be contained and retried by the
     * adapter; close may throw only for an invariant violation.
     */
    interface PreparedSubmission extends AutoCloseable {

        void submitBatch(
                Command command,
                BiConsumer<ScheduledRequest, SlotDeliveryPort.Completion> observer);

        /** Resolve an unused preparation without changing business outcome. */
        @Override
        void close();
    }

    /** Exact canonical batch submitted at the transport boundary. */
    record Command(
            List<ScheduledRequest> exactItems,
            long batchId,
            long predictedMs,
            DeliveryMetadata metadata) {

        public Command {
            exactItems = List.copyOf(exactItems);
            if (exactItems.isEmpty()) {
                throw new IllegalArgumentException("batch cannot be empty");
            }
            if (batchId <= 0L) {
                throw new IllegalArgumentException("batchId must be positive");
            }
            if (predictedMs < 0L) {
                throw new IllegalArgumentException(
                        "predictedMs must be non-negative");
            }
        }
    }
}
