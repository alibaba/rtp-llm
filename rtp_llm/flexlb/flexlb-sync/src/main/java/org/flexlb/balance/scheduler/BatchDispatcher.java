package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;

import java.util.List;
import java.util.Objects;

/**
 * Owns the bounded local handoff used to start one EnqueueBatch RPC.
 *
 * <p>A batch may leave the worker's ACTIVE queue only after
 * {@link #tryReserveSubmission()} returns a {@link SubmissionReserved}. The
 * returned permit represents an executor task which the dispatcher has already
 * accepted. Calling {@link SubmissionPermit#submit} therefore cannot fail
 * because the local dispatch executor is full.
 *
 * <p>The permit covers only the first local dispatch task: request construction,
 * RPC invocation, and completion-observer registration. RPC completion is a
 * later transport event and is not an admission resource.
 */
public interface BatchDispatcher {

    /** Try to reserve one exact local submission task without blocking. */
    SubmissionReservationResult tryReserveSubmission();

    sealed interface SubmissionReservationResult permits SubmissionReserved,
            SubmissionCapacityUnavailable, SubmissionAdmissionFailed {
    }

    record SubmissionReserved(SubmissionPermit permit)
            implements SubmissionReservationResult {
        public SubmissionReserved {
            Objects.requireNonNull(permit, "permit");
        }
    }

    record SubmissionCapacityUnavailable(
            DeliveryCapacityAdmission.CapacityAvailability availability)
            implements SubmissionReservationResult {
        public SubmissionCapacityUnavailable {
            Objects.requireNonNull(availability, "availability");
        }
    }

    record SubmissionAdmissionFailed(Throwable cause)
            implements SubmissionReservationResult {
        public SubmissionAdmissionFailed {
            Objects.requireNonNull(cause, "cause");
        }
    }

    /**
     * Exact ownership of one dispatcher task accepted before batch admission.
     * Exactly one of {@link #submit} or {@link #release} may resolve the permit.
     * A successful {@code submit} return transfers that task to transport
     * ownership. If {@code submit} throws, it must leave the task unsubmitted
     * and releasable; it must not start an RPC.
     */
    interface SubmissionPermit {

        void submit(List<BatchItem> items,
                    PrefillEndpoint prefillEndpoint,
                    long batchId,
                    long predictedMs,
                    String reason,
                    DispatchCallback callback);

        /** Release an accepted task which never became a batch submission. */
        void release();
    }
}
