package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Delivers a prepared decision group through the existing EnqueueBatch RPC. */
final class BatchEnqueueDelivery {

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

    BatchDispatcher.SubmissionReservationResult tryReserveSubmission() {
        return dispatcher.tryReserveSubmission();
    }

    /**
     * Prepare a transport submission whose callbacks stay deferred until the
     * caller explicitly opens the callback gate.
     *
     * <p>The priority scheduler uses this two-step form while it holds its
     * delivery fence around the final lifecycle-to-transport handoff. The
     * dispatcher task was accepted during admission. Callback deferral prevents
     * an immediately completed transport from running response continuations
     * inside the scheduler-wide fence.</p>
     */
    Submission prepare(Plan plan,
                       BatchDispatcher.SubmissionPermit submissionPermit,
                       DecisionDelivery.Callback callback) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(submissionPermit, "submissionPermit");
        Objects.requireNonNull(callback, "callback");
        return new Submission(submissionPermit, plan, callback);
    }

    /** One allocation owns both the transport invocation and its callback gate. */
    static final class Submission implements DispatchCallback {
        private final BatchDispatcher.SubmissionPermit submissionPermit;
        private final Plan plan;
        private final DecisionDelivery.Callback callback;
        private boolean permitResolved;
        private volatile boolean deferCallbacks = true;
        /** Allocated only when a dispatcher invokes a callback before submit returns. */
        private List<CallbackEvent> deferredCallbacks;

        private Submission(BatchDispatcher.SubmissionPermit submissionPermit,
                           Plan plan,
                           DecisionDelivery.Callback callback) {
            this.submissionPermit = submissionPermit;
            this.plan = plan;
            this.callback = callback;
        }

        synchronized void submit() {
            if (permitResolved) {
                throw new IllegalStateException(
                        "batch submission permit was already resolved");
            }
            submissionPermit.submit(
                    plan.items(),
                    plan.prefillEndpoint(),
                    plan.batchId(),
                    plan.estimatedPrefillMs(),
                    plan.decisionReason(),
                    this);
            permitResolved = true;
        }

        synchronized void releaseUnsubmittedPermit() {
            if (permitResolved) {
                return;
            }
            permitResolved = true;
            submissionPermit.release();
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
