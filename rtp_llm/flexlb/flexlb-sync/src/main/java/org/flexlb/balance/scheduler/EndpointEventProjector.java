package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;

/**
 * Total projection boundary from endpoint-owned facts into exact RequestSlot
 * transitions. Endpoint accounting is already committed before this class is
 * called; stale generations are legal no-ops and are never queued for replay.
 */
@Component
public final class EndpointEventProjector {

    private final RequestRegistry scheduler;

    public EndpointEventProjector(RequestRegistry scheduler) {
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
    }

    public void onPrefillStatus(
            PrefillEndpoint source,
            RoleType role,
            List<PrefillState.WorkerStatusFact> facts) {
        projectPrefillStatus(source, role, facts);
    }

    public void onDecodeStatus(
            DecodeEndpoint source,
            List<DecodeEndpoint.WorkerStatusFact> facts) {
        projectDecodeStatus(source, facts);
    }

    public void onEncoderStatus(EncoderEndpoint source,
                                List<WorkerStatus.TaskObservation> running,
                                List<WorkerStatus.TaskObservation> finished) {
        for (WorkerStatus.TaskObservation task : running) {
            try {
                scheduler.processEncoderStatus(source, task, false);
            } catch (Throwable failure) {
                logErrorNoFail("Encoder status fact projection isolated: request_id={}",
                        task.requestId(), failure);
            }
        }
        for (WorkerStatus.TaskObservation task : finished) {
            try {
                scheduler.processEncoderStatus(source, task, true);
            } catch (Throwable failure) {
                logErrorNoFail("Encoder finished fact projection isolated: request_id={}",
                        task.requestId(), failure);
            }
        }
        onEncoderCapacityChanged();
    }

    public void onEncoderCapacityChanged() {
        try {
            scheduler.encoderCapacityChanged();
        } catch (Throwable failure) {
            logErrorNoFail("Encoder capacity notification isolated", failure);
        }
    }

    public void onEncoderGenerationRetired(EncoderEndpoint source, List<String> requestIds) {
        for (String requestId : requestIds) {
            try {
                scheduler.projectEncoderRetirement(source, requestId);
            } catch (Throwable failure) {
                logErrorNoFail("Encoder retirement projection isolated: request_id={}",
                        requestId, failure);
            }
        }
    }

    public void onPrefillGenerationRetired(
            PrefillEndpoint endpoint,
            List<ScheduledRequest> ownedItems) {
        if (endpoint == null || ownedItems == null) {
            return;
        }
        try {
            projectPrefillRetirementFacts(endpoint, ownedItems);
        } catch (Throwable failure) {
            logErrorNoFail("Endpoint event projection isolated: event={}",
                    "prefill retirement", failure);
        }
    }

    public void onDecodeGenerationRetired(
            DecodeEndpoint endpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations) {
        if (endpoint == null || ownedReservations == null) {
            return;
        }
        try {
            projectDecodeRetirementFacts(endpoint, ownedReservations);
        } catch (Throwable failure) {
            logErrorNoFail("Endpoint event projection isolated: event={}",
                    "decode retirement", failure);
        }
    }

    public void onQueuedItemExpired(ScheduledRequest exactItem) {
        scheduler.onQueuedItemExpired(exactItem);
    }

    public void onQueuedItemPreempted(ScheduledRequest victim, ScheduledRequest incoming) {
        try {
            scheduler.onQueuedItemPreempted(victim, incoming);
        } catch (Throwable failure) {
            logErrorNoFail("Queued preemption projection failed: request_id={}", victim.requestId(), failure);
        }
    }

    public void onQueueOfferFailure(
            ScheduledRequest exactItem, Throwable cause) {
        scheduler.onQueueOfferFailure(exactItem, cause);
    }

    public void onPreparedDeliveryFailure(
            ScheduledRequest exactItem, Throwable cause) {
        scheduler.failDeliveryPreparation(exactItem, cause);
    }

    private void projectPrefillStatus(
            PrefillEndpoint source,
            RoleType role,
            List<PrefillState.WorkerStatusFact> facts) {
        for (PrefillState.WorkerStatusFact fact : facts) {
            try {
                scheduler.processPrefillStatus(source, role, fact);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Prefill status fact projection isolated: request_id={} engine={}",
                        fact.item().requestId(), source.getIp(),
                        failure);
            }
        }
    }

    private void projectDecodeStatus(
            DecodeEndpoint source,
            List<DecodeEndpoint.WorkerStatusFact> facts) {
        for (DecodeEndpoint.WorkerStatusFact fact : facts) {
            try {
                scheduler.processDecodeStatus(source, fact);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Decode status fact projection isolated: request_id={} engine={}",
                        fact.reservation().requestId(),
                        source.getIp(), failure);
            }
        }
    }

    private void projectPrefillRetirementFacts(
            PrefillEndpoint retiredEndpoint,
            List<ScheduledRequest> ownedItems) {
        for (int index = 0; index < ownedItems.size(); index++) {
            ScheduledRequest exactItem = ownedItems.get(index);
            try {
                scheduler.projectPrefillRetirementItem(
                        retiredEndpoint, exactItem);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Prefill retirement item projection isolated: request_id={} engine={}",
                        exactItem == null ? -1 : exactItem.requestId(),
                        retiredEndpoint.getIp(), failure);
            }
        }
    }

    private void projectDecodeRetirementFacts(
            DecodeEndpoint retiredEndpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations) {
        for (int index = 0; index < ownedReservations.size(); index++) {
            DecodeEndpoint.ReservationHandle reservation =
                    ownedReservations.get(index);
            try {
                scheduler.projectDecodeRetirementReservation(
                        retiredEndpoint, reservation);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Decode retirement reservation projection isolated: "
                                + "request_id={} generation={}",
                        reservation.requestId(),
                        reservation.endpointGenerationId(),
                        failure);
            }
        }
    }

    private static void logErrorNoFail(String format, Object... arguments) {
        try {
            Logger.error(format, arguments);
        } catch (Throwable ignoredDiagnosticFailure) {
            // Continue projecting the remaining exact facts.
        }
    }

}
