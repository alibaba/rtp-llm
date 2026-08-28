package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryItem;

import java.util.List;

/** Exact endpoint-owned fact delivered synchronously to the request runtime. */
public sealed interface EndpointEvent {

    record StatusReduced(EndpointStatusReduction reduction)
            implements EndpointEvent {
    }

    /** Preallocated publication shell bound after the ledger retirement commit. */
    final class PrefillGenerationRetired implements EndpointEvent {
        private final PrefillEndpoint endpoint;
        private List<DeliveryItem> ownedItems;

        PrefillGenerationRetired(PrefillEndpoint endpoint) {
            this.endpoint = endpoint;
        }

        /** Bind the ledger-owned immutable facts without allocating or copying. */
        void bindOwnedItems(List<DeliveryItem> exactOwnedItems) {
            ownedItems = exactOwnedItems;
        }

        public PrefillEndpoint endpoint() {
            return endpoint;
        }

        public List<DeliveryItem> ownedItems() {
            return ownedItems;
        }
    }

    record DecodeGenerationRetired(
            DecodeEndpoint endpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations)
            implements EndpointEvent {
        public DecodeGenerationRetired {
            ownedReservations = List.copyOf(ownedReservations);
        }
    }
}
