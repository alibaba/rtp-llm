package org.flexlb.balance.scheduler;

/** Publishes scheduler-prepared work without owning scheduler state. */
interface DecisionDelivery<T> {

    void deliver(T delivery, Callback callback);

    /** Receives the per-request outcome of a delivery attempt. */
    interface Callback {

        /** The mode-specific delivery boundary accepted this request. */
        void onDelivered(BatchItem item);

        /** Delivery failed before ownership became ambiguous. */
        void onFailure(BatchItem item, Throwable error);

        /** Delivery timed out before a trustworthy outcome was observed. */
        default void onTimeout(BatchItem item, Throwable error) {
            onFailure(item, error);
        }

        /** Delivery started, but its externally visible outcome is uncertain. */
        default void onUncertain(BatchItem item, Throwable error) {
            onTimeout(item, error);
        }
    }
}
