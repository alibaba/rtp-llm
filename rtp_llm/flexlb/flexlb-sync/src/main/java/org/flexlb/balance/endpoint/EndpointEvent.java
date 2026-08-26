package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.DeliveryItem;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/** Exact endpoint-owned fact delivered synchronously to the request runtime. */
public sealed interface EndpointEvent {

    record StatusReduced(EndpointStatusReduction reduction)
            implements EndpointEvent {
        public StatusReduced {
            Objects.requireNonNull(reduction, "reduction");
        }
    }

    /** Preallocated publication shell bound after the ledger retirement commit. */
    final class PrefillGenerationRetired implements EndpointEvent {
        private final PrefillEndpoint endpoint;
        private List<DeliveryItem> ownedItems;

        PrefillGenerationRetired(PrefillEndpoint endpoint) {
            this.endpoint = Objects.requireNonNull(endpoint, "endpoint");
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
            Objects.requireNonNull(endpoint, "endpoint");
            Objects.requireNonNull(
                    ownedReservations, "ownedReservations");
            for (DecodeEndpoint.ReservationHandle reservation
                    : ownedReservations) {
                Objects.requireNonNull(
                        reservation, "owned reservation");
            }
            ownedReservations = List.copyOf(ownedReservations);
            if (new HashSet<>(ownedReservations).size()
                    != ownedReservations.size()) {
                throw new IllegalArgumentException(
                        "Decode retirement reservations must be unique");
            }
            long generationId = endpoint.getStatus().getGenerationId();
            for (DecodeEndpoint.ReservationHandle reservation
                    : ownedReservations) {
                if (reservation.endpointGenerationId() != generationId) {
                    throw new IllegalArgumentException(
                            "Decode retirement reservation belongs to another generation");
                }
            }
        }
    }
}
