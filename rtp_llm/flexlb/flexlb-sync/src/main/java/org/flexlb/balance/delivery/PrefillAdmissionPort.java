package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;
import java.util.OptionalLong;

/**
 * Composite endpoint admission required by delivery strategies.
 *
 * <p>Lease, permit, generation-fence, and cleanup implementation details are
 * hidden behind group-scoped prepared and committed capabilities.
 */
public interface PrefillAdmissionPort {

    /** Begin an empty admission transaction scoped by the first candidate. */
    CapacityBoundary.Attempt<PreparedAdmission> tryBegin(
            ScheduledRequest firstCandidate);

    /** Atomic admission transaction for one exact ordered prefix. */
    interface PreparedAdmission extends AutoCloseable {

        /**
         * Transport correlation allocated by this admission, when required.
         * The value is stable for the lifetime of this capability.
         */
        OptionalLong correlationId();

        /** Prepare the next exact member; acceptance returns that same object. */
        CapacityBoundary.Attempt<ScheduledRequest> tryAppend(
                ScheduledRequest exactNextItem,
                long predictedMs);

        /**
         * Commit the exact prepared identity sequence while the queue lock is
         * held. Success moves ownership to the returned capability. A later
         * try-with-resources close of this prepared object must be a no-op.
         * Repeated commit or append after commit must throw.
         */
        CommittedAdmission commitPreparedUnderLock(
                List<ScheduledRequest> exactItems,
                long predictedMs);

        /**
         * Roll back everything prepared unless ownership already moved.
         * Every rollback leaf is attempted and failures are aggregated.
         */
        @Override
        void close();
    }

    /**
     * Opaque post-commit endpoint handoff for one exact admitted group.
     *
     * <p>Each member may transfer exactly once. Unknown identities, repeated
     * transfer, or transfer after close are illegal and must throw. Closing
     * releases only capabilities that have not already moved to an endpoint.
     * Post-commit cleanup failures are isolated from the terminal business
     * result and are never retried by this capability. Transfer is a local,
     * synchronous leaf operation: implementations may acquire endpoint-local
     * locks and publish capacity signals in their documented order, but must
     * not perform I/O, await external completion, or call back into a request
     * slot, delivery lifecycle, or user code.
     */
    interface CommittedAdmission extends AutoCloseable {

        boolean transferToEndpoint(ScheduledRequest exactItem);

        @Override
        void close();
    }
}
