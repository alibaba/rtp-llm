package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.service.monitor.StatusFailureReporter;
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
    private final StatusFailureReporter failureReporter;

    public EndpointEventProjector(RequestRegistry scheduler, StatusFailureReporter failureReporter) {
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
        this.failureReporter = Objects.requireNonNull(failureReporter, "failureReporter");
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
                reportStatusFailure(role, failure);
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
                reportStatusFailure(RoleType.DECODE, failure);
                logErrorNoFail(
                        "Decode status fact projection isolated: request_id={} engine={}",
                        fact.reservation().requestId(),
                        source.getIp(), failure);
            }
        }
    }

    private void reportStatusFailure(RoleType role, Throwable failure) {
        try {
            GrpcFailure.Analysis analysis = GrpcFailure.analyze(failure);
            failureReporter.report(role, "worker_status", Stage.STATUS_PROJECTION_FAILED,
                    analysis.confirmedLocalFailure() ? analysis.kind().name() : "STATUS_PROJECTION_FAILED",
                    analysis);
        } catch (RuntimeException | Error ignoredDiagnosticFailure) {
            // Continue projecting the remaining exact facts.
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
