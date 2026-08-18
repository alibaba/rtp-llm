package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Delivers a prepared decision group through the existing EnqueueBatch RPC. */
final class BatchEnqueueDelivery implements DecisionDelivery<BatchEnqueueDelivery.Plan> {

    private static final Logger LOGGER = LoggerFactory.getLogger(BatchEnqueueDelivery.class);

    /** Data required by one EnqueueBatch call. */
    record Plan(
            List<BatchItem> items,
            PrefillEndpoint prefillEndpoint,
            long batchId,
            long estimatedPrefillMs,
            String decisionReason) {

        Plan {
            Objects.requireNonNull(items, "items");
            Objects.requireNonNull(prefillEndpoint, "prefillEndpoint");
            Objects.requireNonNull(decisionReason, "decisionReason");
            if (items.isEmpty()) {
                throw new IllegalArgumentException(
                        "batch-enqueue delivery requires at least one request");
            }
            if (batchId <= 0) {
                throw new IllegalArgumentException(
                        "batch-enqueue delivery requires a positive batch id");
            }
            for (BatchItem item : items) {
                Objects.requireNonNull(item, "items must not contain null");
                if (item.deliveryMode() != DeliveryMode.BATCH_ENQUEUE) {
                    throw new IllegalArgumentException(
                            "batch-enqueue delivery contains a route-decision request");
                }
                if (item.prefillEp() != prefillEndpoint) {
                    throw new IllegalArgumentException(
                            "batch-enqueue delivery contains requests for different Prefill endpoints");
                }
            }
        }
    }

    private final BatchDispatcher dispatcher;

    BatchEnqueueDelivery(BatchDispatcher dispatcher) {
        this.dispatcher = Objects.requireNonNull(dispatcher, "dispatcher");
    }

    @Override
    public void deliver(Plan plan, Callback callback) {
        Submission submission = prepare(plan, callback);
        try {
            submission.submit();
        } finally {
            submission.releaseCallbacks();
        }
    }

    /**
     * Prepare a transport submission whose callbacks stay deferred until the
     * caller explicitly opens the callback gate.
     *
     * <p>The priority scheduler uses this two-step form while it holds its
     * delivery fence around the final commit-to-transport handoff. A bounded
     * dispatcher may reject synchronously and invoke its failure callback on
     * the submitting thread; deferring that callback prevents response-future
     * continuations from running inside the scheduler-wide fence. Callbacks
     * arriving after the gate opens take one volatile read and otherwise retain
     * the existing direct asynchronous path.</p>
     */
    Submission prepare(Plan plan, Callback callback) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(callback, "callback");
        return new Submission(dispatcher, plan, callback);
    }

    /** One allocation owns both the transport invocation and its callback gate. */
    static final class Submission implements DispatchCallback {
        private final BatchDispatcher dispatcher;
        private final Plan plan;
        private final Callback callback;
        private volatile boolean deferCallbacks = true;
        /** Allocated only when a dispatcher invokes a callback before submit returns. */
        private List<CallbackEvent> deferredCallbacks;

        private Submission(BatchDispatcher dispatcher, Plan plan, Callback callback) {
            this.dispatcher = dispatcher;
            this.plan = plan;
            this.callback = callback;
        }

        void submit() {
            dispatcher.dispatch(
                    plan.items(),
                    plan.prefillEndpoint(),
                    plan.batchId(),
                    plan.estimatedPrefillMs(),
                    plan.decisionReason(),
                    this);
        }

        void releaseCallbacks() {
            List<CallbackEvent> callbacks;
            synchronized (this) {
                if (!deferCallbacks) {
                    return;
                }
                deferCallbacks = false;
                callbacks = deferredCallbacks;
                deferredCallbacks = null;
            }
            if (callbacks == null) {
                return;
            }
            for (CallbackEvent event : callbacks) {
                try {
                    invoke(event.kind(), event.item(), event.batchId(), event.error());
                } catch (Throwable callbackFailure) {
                    // The transport already accepted responsibility for every
                    // item. One scheduler callback must not suppress later item
                    // outcomes in the same synchronously rejected batch.
                    LOGGER.error("Deferred batch-delivery callback failed request_id={} kind={}",
                            event.item().requestId(), event.kind(), callbackFailure);
                }
            }
        }

        @Override
        public void onSuccess(BatchItem item, long batchId) {
            publish(CallbackKind.SUCCESS, item, batchId, null);
        }

        @Override
        public void onFailure(BatchItem item, Throwable error) {
            publish(CallbackKind.FAILURE, item, plan.batchId(), error);
        }

        @Override
        public void onTimeout(BatchItem item, Throwable error) {
            publish(CallbackKind.TIMEOUT, item, plan.batchId(), error);
        }

        @Override
        public void onDispatchUncertain(BatchItem item, long batchId, Throwable error) {
            publish(CallbackKind.UNCERTAIN, item, batchId, error);
        }

        private void publish(CallbackKind kind,
                             BatchItem item,
                             long batchId,
                             Throwable error) {
            if (!deferCallbacks) {
                invoke(kind, item, batchId, error);
                return;
            }
            synchronized (this) {
                if (deferCallbacks) {
                    if (deferredCallbacks == null) {
                        deferredCallbacks = new ArrayList<>();
                    }
                    deferredCallbacks.add(new CallbackEvent(kind, item, batchId, error));
                    return;
                }
            }
            invoke(kind, item, batchId, error);
        }

        private void invoke(CallbackKind kind,
                            BatchItem item,
                            long batchId,
                            Throwable error) {
            switch (kind) {
                case SUCCESS -> {
                    if (batchId != plan.batchId()) {
                        callback.onUncertain(item, batchIdMismatch(batchId, null));
                    } else {
                        callback.onDelivered(item);
                    }
                }
                case FAILURE -> callback.onFailure(item, error);
                case TIMEOUT -> callback.onTimeout(item, error);
                case UNCERTAIN -> callback.onUncertain(item,
                        batchId == plan.batchId() ? error : batchIdMismatch(batchId, error));
            }
        }

        private IllegalStateException batchIdMismatch(long actualBatchId, Throwable cause) {
            return new IllegalStateException(
                    "EnqueueBatch callback batch id " + actualBatchId
                            + " does not match delivery plan batch id " + plan.batchId(),
                    cause);
        }

        private enum CallbackKind {
            SUCCESS,
            FAILURE,
            TIMEOUT,
            UNCERTAIN
        }

        private record CallbackEvent(CallbackKind kind,
                                     BatchItem item,
                                     long batchId,
                                     Throwable error) {
        }
    }
}
