package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EncoderEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.prefill.WorkerStatusFact;
import org.flexlb.balance.schedulev2.prefill.PrefillQueueRegistry;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Optional;

/**
 * 节点完成状态提交后，向对应的调度链路交付同一次 WorkerStatus。
 * V2 Prefill 队列消费活跃及完成明细；旧调度器消费节点账本产生的请求事件。
 */
@Component
public final class EndpointEventHandler {

    private final RequestRegistry scheduler;
    private final PrefillQueueRegistry prefillQueueRegistry;

    public EndpointEventHandler(RequestRegistry scheduler, PrefillQueueRegistry prefillQueueRegistry) {
        this.scheduler = scheduler;
        this.prefillQueueRegistry = prefillQueueRegistry;
    }

    public void onPrefillStatus(PrefillEndpoint prefillEndpoint,
                                WorkerStatus.StatusObservation statusObservation,
                                List<WorkerStatusFact> facts) {
        handlePrefillStatus(prefillEndpoint, statusObservation, facts);
    }

    public void onDecodeStatus(
            DecodeEndpoint source,
            List<DecodeEndpoint.WorkerStatusFact> facts) {
        handleDecodeStatus(source, facts);
    }

    public void onEncoderStatus(EncoderEndpoint source,
                                List<WorkerStatus.TaskObservation> running,
                                List<WorkerStatus.TaskObservation> finished) {
        for (WorkerStatus.TaskObservation task : running) {
            try {
                scheduler.processEncoderStatus(source, task, false);
            } catch (Throwable failure) {
                logErrorNoFail("Encoder status fact handling isolated: request_id={}",
                        task.requestId(), failure);
            }
        }
        for (WorkerStatus.TaskObservation task : finished) {
            try {
                scheduler.processEncoderStatus(source, task, true);
            } catch (Throwable failure) {
                logErrorNoFail("Encoder finished fact handling isolated: request_id={}",
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
                scheduler.applyEncoderRetirement(source, requestId);
            } catch (Throwable failure) {
                logErrorNoFail("Encoder retirement handling isolated: request_id={}",
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
            handlePrefillRetirement(endpoint, ownedItems);
        } catch (Throwable failure) {
            logErrorNoFail("Endpoint event handling isolated: event={}",
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
            handleDecodeRetirement(endpoint, ownedReservations);
        } catch (Throwable failure) {
            logErrorNoFail("Endpoint event handling isolated: event={}",
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
            logErrorNoFail("Queued preemption handling failed: request_id={}", victim.requestId(), failure);
        }
    }

    public void onQueuedWorkerStopped(ScheduledRequest itemFromStoppedWorker) {
        scheduler.onQueuedWorkerStopped(itemFromStoppedWorker);
    }

    public void onQueueOfferFailure(
            ScheduledRequest exactItem, Throwable cause) {
        scheduler.onQueueOfferFailure(exactItem, cause);
    }

    public void onPreparedDeliveryFailure(
            ScheduledRequest exactItem, Throwable cause) {
        scheduler.failDeliveryPreparation(exactItem, cause);
    }

    private void handlePrefillStatus(PrefillEndpoint source,
                                     WorkerStatus.StatusObservation statusObservation,
                                     List<WorkerStatusFact> facts) {
        Optional.ofNullable(prefillQueueRegistry.find(source))
                .ifPresent(queue -> queue.handlePrefillStatus(statusObservation));
        for (WorkerStatusFact fact : facts) {
            try {
                scheduler.processPrefillStatus(source, statusObservation.role(), fact);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Prefill status fact handling isolated: request_id={} engine={}",
                        fact.item().requestId(), source.getIp(),
                        failure);
            }
        }
    }

    private void handleDecodeStatus(
            DecodeEndpoint source,
            List<DecodeEndpoint.WorkerStatusFact> facts) {
        for (DecodeEndpoint.WorkerStatusFact fact : facts) {
            try {
                scheduler.processDecodeStatus(source, fact);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Decode status fact handling isolated: request_id={} engine={}",
                        fact.reservation().requestId(),
                        source.getIp(), failure);
            }
        }
    }

    private void handlePrefillRetirement(
            PrefillEndpoint retiredEndpoint,
            List<ScheduledRequest> ownedItems) {
        for (int index = 0; index < ownedItems.size(); index++) {
            ScheduledRequest exactItem = ownedItems.get(index);
            try {
                scheduler.applyPrefillRetirementItem(
                        retiredEndpoint, exactItem);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Prefill retirement item handling isolated: request_id={} engine={}",
                        exactItem == null ? -1 : exactItem.requestId(),
                        retiredEndpoint.getIp(), failure);
            }
        }
    }

    private void handleDecodeRetirement(
            DecodeEndpoint retiredEndpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations) {
        for (int index = 0; index < ownedReservations.size(); index++) {
            DecodeEndpoint.ReservationHandle reservation =
                    ownedReservations.get(index);
            try {
                scheduler.applyDecodeRetirementReservation(
                        retiredEndpoint, reservation);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Decode retirement reservation handling isolated: "
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
            // Continue handling the remaining exact facts.
        }
    }

}
