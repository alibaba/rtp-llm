package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.config.DispatcherConfig;

import java.util.Objects;

/** Dispatcher-owned conversion from endpoint state to publication credits. */
sealed interface DeliveryCreditPolicy {

    int availableCredits(PrefillState state);

    static DeliveryCreditPolicy from(
            DispatcherConfig dispatcher,
            int maximumQueuedRequests,
            int maximumRequestsPerDecision) {
        Objects.requireNonNull(dispatcher, "dispatcher");
        return switch (dispatcher.getType()) {
            case NON_BATCH -> new NonBatch(
                    unlimitedAsZero(
                            dispatcher.getMaxInflightRequestsPerPrefillWorker()));
            case BATCH -> new Batch(
                    unlimitedAsZero(
                            dispatcher.getMaxInflightBatchesPerPrefillWorker()),
                    maximumQueuedRequests,
                    maximumRequestsPerDecision);
        };
    }

    private static int unlimitedAsZero(Integer configured) {
        return configured == null ? 0 : configured;
    }

    record NonBatch(int maximumInflightRequests)
            implements DeliveryCreditPolicy {

        @Override
        public int availableCredits(PrefillState state) {
            return state.availableRoutePublicationCredits(
                    maximumInflightRequests);
        }
    }

    record Batch(
            int maximumInflightBatches,
            int maximumQueuedRequests,
            int maximumRequestsPerDecision)
            implements DeliveryCreditPolicy {

        @Override
        public int availableCredits(PrefillState state) {
            return state.availableBatchPublicationCredits(
                    maximumInflightBatches,
                    maximumRequestsPerDecision,
                    maximumQueuedRequests);
        }
    }
}
