package org.flexlb.balance.eviction;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime.QueueSnapshot;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/** Canonical placement boundary for eviction-backed request admission. */
public interface EvictionPlacementPort {

    PreparedDecodePlacement prepareDecodePlacement(
            BalanceContext context,
            CompletableFuture<Response> future,
            DecodeEndpoint endpoint);

    PrefillEvictionAdmission preparePrefillEviction(
            BalanceContext context,
            CompletableFuture<Response> future);

    sealed interface DecodePlacement {
        record Committed() implements DecodePlacement {
        }

        record Failed(AdmissionFailure failure) implements DecodePlacement {
        }
    }

    interface PreparedDecodePlacement extends AutoCloseable {

        /** Seal the exact Prefill seat immediately before Decode victim PNR. */
        boolean seal();

        DecodePlacement commit(
                DecodeEndpoint.ReservationHandle reservation,
                AdmissionMutation exactMutation);

        @Override
        void close();
    }

    interface PrefillEvictionAdmission extends AutoCloseable {
        PriorityRequestEnvelope envelope();

        QueueSnapshot queueSnapshot();

        PrefillEvictionCommit commit(
                List<DeliveryItem> exactVictims,
                AdmissionMutation exactMutation);

        @Override
        void close();
    }

    enum PrefillEvictionStatus {
        COMMITTED,
        CONFLICT,
        DECLINED
    }

    record PrefillEvictionCommit(
            PrefillEvictionStatus status,
            List<DeliveryItem> removed) {
        public PrefillEvictionCommit {
            assert status != null : "missing Prefill eviction status";
            removed = List.copyOf(removed);
            if (status != PrefillEvictionStatus.COMMITTED
                    && !removed.isEmpty()) {
                throw new IllegalArgumentException(
                        "non-committed Prefill replacement cannot remove victims");
            }
            if (status == PrefillEvictionStatus.COMMITTED
                    && removed.isEmpty()) {
                throw new IllegalArgumentException(
                        "committed Prefill replacement requires exact victims");
            }
        }
    }
}
