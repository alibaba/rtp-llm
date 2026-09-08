package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.config.DispatcherConfig;

import java.util.Objects;

/** Dispatcher-owned conversion from endpoint state to publication credits. */
sealed interface DeliveryCreditPolicy {

    PrefillState.PublicationCapacity capacity(PrefillState state,
                                             int maximumQueuedRequests,
                                             int maximumRequestsPerDecision);

    static DeliveryCreditPolicy from(
            DispatcherConfig dispatcher) {
        Objects.requireNonNull(dispatcher, "dispatcher");
        return switch (dispatcher.getType()) {
            case NON_BATCH -> new NonBatch(
                    unlimitedAsZero(
                            dispatcher.getMaxInflightRequestsPerPrefillWorker()));
            case BATCH -> new Batch(
                    unlimitedAsZero(
                            dispatcher.getMaxInflightBatchesPerPrefillWorker()));
        };
    }

    private static int unlimitedAsZero(Integer configured) {
        return configured == null ? 0 : configured;
    }

    record NonBatch(int maximumInflightRequests)
            implements DeliveryCreditPolicy {

        @Override
        public PrefillState.PublicationCapacity capacity(PrefillState state,
                                                        int maximumQueuedRequests,
                                                        int maximumRequestsPerDecision) {
            return state.routePublicationCapacity(
                    maximumInflightRequests, maximumQueuedRequests);
        }
    }

    record Batch(int maximumInflightBatches)
            implements DeliveryCreditPolicy {

        @Override
        public PrefillState.PublicationCapacity capacity(PrefillState state,
                                                        int maximumQueuedRequests,
                                                        int maximumRequestsPerDecision) {
            return state.batchPublicationCapacity(
                    maximumInflightBatches,
                    maximumRequestsPerDecision,
                    maximumQueuedRequests);
        }
    }
}
