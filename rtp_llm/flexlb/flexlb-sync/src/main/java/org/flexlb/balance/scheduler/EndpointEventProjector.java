package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointEventSink;
import org.flexlb.balance.endpoint.EndpointStatusReduction;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.dao.loadbalance.StrategyErrorType;
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
final class EndpointEventProjector implements EndpointEventSink {
    private final RequestRegistry scheduler;

    EndpointEventProjector(RequestRegistry scheduler) {
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
    }

    @Override
    public void onStatusReduced(EndpointStatusReduction reduction) {
        if (reduction == null) {
            return;
        }
        try {
            projectStatusReduction(reduction);
        } catch (Throwable failure) {
            logEventProjectionFailureNoFail("status", failure);
        }
    }

    @Override
    public void onPrefillGenerationRetired(
            PrefillEndpoint endpoint,
            List<ScheduledRequest> ownedItems) {
        if (endpoint == null || ownedItems == null) {
            return;
        }
        try {
            projectPrefillRetirementFacts(endpoint, ownedItems);
        } catch (Throwable failure) {
            logEventProjectionFailureNoFail("prefill retirement", failure);
        }
    }

    @Override
    public void onDecodeGenerationRetired(
            DecodeEndpoint endpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations) {
        if (endpoint == null || ownedReservations == null) {
            return;
        }
        try {
            projectDecodeRetirementFacts(endpoint, ownedReservations);
        } catch (Throwable failure) {
            logEventProjectionFailureNoFail("decode retirement", failure);
        }
    }

    @Override
    public void onQueuedItemExpired(ScheduledRequest exactItem) {
        scheduler.onQueuedItemExpired(exactItem);
    }

    @Override
    public void onQueueOfferFailure(
            ScheduledRequest exactItem, Throwable cause) {
        scheduler.onQueueOfferFailure(exactItem, cause);
    }

    @Override
    public void onPreparedDeliveryFailure(
            ScheduledRequest exactItem, Throwable cause) {
        scheduler.onPreparedDeliveryFailure(exactItem, cause);
    }

    private void projectStatusReduction(EndpointStatusReduction reduction) {
        long observedAtMs = System.currentTimeMillis();
        switch (reduction) {
            case EndpointStatusReduction.None ignored -> { }
            case PrefillEndpoint.StatusReduction prefill ->
                    projectPrefillStatus(prefill, observedAtMs);
            case DecodeEndpoint.StatusReduction decode ->
                    projectDecodeStatus(decode, observedAtMs);
        }
    }

    private void projectPrefillStatus(
            PrefillEndpoint.StatusReduction reduction,
            long observedAtMs) {
        for (PrefillState.WorkerStatusFact fact : reduction.facts()) {
            try {
                switch (fact.kind()) {
                    case ACTIVE -> projectPrefillActive(
                            reduction.source(), fact, observedAtMs);
                    case COMPLETED, FAILED, PRIORITY_CANCELED ->
                            projectPrefillTerminal(reduction, fact);
                }
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Prefill status fact projection isolated: request_id={} engine={}",
                        fact.item().requestId(), reduction.source().getIp(),
                        failure);
            }
        }
    }

    private void projectPrefillActive(
            PrefillEndpoint source,
            PrefillState.WorkerStatusFact fact,
            long observedAtMs) {
        ScheduledRequest item = fact.item();
        RequestSlot slot = scheduler.requestSlot(item.requestId());
        if (slot == null) {
            return;
        }
        RequestRegistry.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)
                    || !slot.ownsPrefillFact(source, item)) {
                return;
            }
            slot.observeWorkerStatus(observedAtMs);
            work = scheduler.materializePreemptionWorkLocked(
                    slot,
                    slot.reducePrefillActive(source, item),
                    null);
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private void projectPrefillTerminal(
            PrefillEndpoint.StatusReduction reduction,
            PrefillState.WorkerStatusFact fact) {
        if (fact.kind() == PrefillState.WorkerStatusFact.Kind.COMPLETED
                && reduction.semantics()
                    != PrefillEndpoint.StatusSemantics.FUSION_TERMINAL) {
            logWarnNoFail(
                    "Ignoring Prefill-stage successful terminal projection: request_id={} engine={}",
                    fact.item().requestId(), reduction.source().getIp());
            return;
        }

        ScheduledRequest item = fact.item();
        RequestSlot slot = scheduler.requestSlot(item.requestId());
        if (slot == null) {
            return;
        }
        RequestRegistry.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)
                    || !slot.ownsPrefillFact(reduction.source(), item)) {
                return;
            }
            WorkerTerminalObservation observation =
                    new WorkerTerminalObservation(
                            WorkerTerminalSource.PREFILL_BACKED,
                            fact.kind()
                                == PrefillState.WorkerStatusFact.Kind.COMPLETED,
                            fact.errorCode());
            DeferredTerminal terminal = DeferredTerminal.worker(observation);
            if (fact.kind()
                    == PrefillState.WorkerStatusFact.Kind.PRIORITY_CANCELED) {
                work = scheduler.materializePreemptionWorkLocked(
                        slot,
                        slot.reducePriorityCanceled(
                                reduction.source(), item),
                        null);
            } else {
                work = scheduler.materializePreemptionWorkLocked(
                        slot,
                        slot.reduceWorkerTerminal(item, terminal),
                        null);
            }
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private void projectDecodeStatus(
            DecodeEndpoint.StatusReduction reduction,
            long observedAtMs) {
        for (DecodeEndpoint.WorkerStatusFact fact : reduction.facts()) {
            try {
                switch (fact.kind()) {
                    case ACTIVE -> projectDecodeActive(
                            reduction.source(), fact, observedAtMs);
                    case ACCEPTED -> projectDecodeAccepted(
                            reduction.source(), fact, observedAtMs);
                    case TERMINAL -> projectDecodeTerminal(
                            reduction.source(), fact);
                }
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Decode status fact projection isolated: request_id={} engine={}",
                        fact.reservation().requestId(),
                        reduction.source().getIp(), failure);
            }
        }
    }

    private void projectDecodeActive(
            DecodeEndpoint source,
            DecodeEndpoint.WorkerStatusFact fact,
            long observedAtMs) {
        RequestSlot slot = scheduler.requestSlot(fact.reservation().requestId());
        if (slot == null) {
            return;
        }
        synchronized (slot) {
            if (scheduler.isCurrentSlot(slot)
                    && slot.ownsDecodeFact(source, fact.reservation())) {
                slot.observeWorkerStatus(observedAtMs);
            }
        }
    }

    private void projectDecodeAccepted(
            DecodeEndpoint source,
            DecodeEndpoint.WorkerStatusFact fact,
            long observedAtMs) {
        RequestSlot slot = scheduler.requestSlot(fact.reservation().requestId());
        if (slot == null) {
            return;
        }
        DecodeAcceptance acceptance;
        RequestRegistry.PreemptionWork work = null;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)
                    || !slot.ownsDecodeFact(source, fact.reservation())) {
                return;
            }
            slot.observeWorkerStatus(observedAtMs);
            acceptance = slot.markDecodeAccepted();
            if (acceptance.acceptedBeforeCancel()
                    && acceptance.releasableFence() != null) {
                work = scheduler.materializePreemptionWorkLocked(
                        slot,
                        slot.reduceDeliveryConfirmed(
                                slot.snapshot().batchId()),
                        null);
            }
        }
        releaseDecodeAcceptance(acceptance, fact.reservation().requestId());
        scheduler.consumePreemptionWork(slot, work);
    }

    private void releaseDecodeAcceptance(
            DecodeAcceptance acceptance,
            long requestId) {
        Throwable failure = null;
        if (acceptance.releasableFence() != null) {
            try {
                acceptance.releasableFence().release();
            } catch (Throwable cleanupFailure) {
                failure = cleanupFailure;
            }
        }
        try {
            scheduler.releaseAdmissionCleanup(acceptance.admissionCleanup());
        } catch (Throwable cleanupFailure) {
            failure = RequestRegistry.appendFailure(
                    failure, cleanupFailure);
        }
        if (failure != null) {
            logErrorNoFail(
                    "Decode acceptance cleanup isolated: request_id={}",
                    requestId, failure);
        }
    }

    private void projectDecodeTerminal(
            DecodeEndpoint source,
            DecodeEndpoint.WorkerStatusFact fact) {
        RequestSlot slot = scheduler.requestSlot(fact.reservation().requestId());
        if (slot == null) {
            return;
        }
        RequestRegistry.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)
                    || !slot.ownsDecodeFact(source, fact.reservation())) {
                return;
            }
            slot.markDecodeTerminalOwned();
            WorkerTerminalObservation observation =
                    new WorkerTerminalObservation(
                            WorkerTerminalSource.DECODE_ENDPOINT_SETTLED,
                            fact.errorCode() == 0L,
                            fact.errorCode());
            DeferredTerminal terminal = DeferredTerminal.worker(observation);
            work = scheduler.materializePreemptionWorkLocked(
                    slot,
                    slot.reduceWorkerTerminal(slot.activeItem(), terminal),
                    null);
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private void projectPrefillRetirementFacts(
            PrefillEndpoint retiredEndpoint,
            List<ScheduledRequest> ownedItems) {
        for (int index = 0; index < ownedItems.size(); index++) {
            ScheduledRequest exactItem = ownedItems.get(index);
            try {
                projectPrefillRetirementItem(
                        retiredEndpoint, exactItem);
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Prefill retirement item projection isolated: request_id={} engine={}",
                        exactItem == null ? -1 : exactItem.requestId(),
                        retiredEndpoint.getIp(), failure);
            }
        }
    }

    private void projectPrefillRetirementItem(
            PrefillEndpoint retiredEndpoint,
            ScheduledRequest exactItem) {
        if (exactItem == null || exactItem.prefillEp() != retiredEndpoint) {
            logErrorNoFail(
                    "Ignoring Prefill retirement item from another generation: request_id={}",
                    exactItem == null ? -1 : exactItem.requestId());
            return;
        }
        ScheduledRequest item = exactItem;
        RequestSlot slot = scheduler.requestSlot(item.requestId());
        if (slot == null) {
            return;
        }
        String detail = "Prefill endpoint generation retired: "
                + retiredEndpoint.ipPort() + "#"
                + retiredEndpoint.getStatus().getGenerationId();
        TerminalAction action;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)) {
                return;
            }
            action = slot.beginPrefillRetirementTerminal(
                    retiredEndpoint,
                    item,
                    owner -> owner.fail(detail),
                    RequestRegistry.buildErrorResponse(
                            StrategyErrorType.BATCH_DISPATCH_FAILED, detail));
        }
        scheduler.submitTerminal(action);
    }

    private void projectDecodeRetirementFacts(
            DecodeEndpoint retiredEndpoint,
            List<DecodeEndpoint.ReservationHandle> ownedReservations) {
        for (int index = 0; index < ownedReservations.size(); index++) {
            DecodeEndpoint.ReservationHandle reservation =
                    ownedReservations.get(index);
            try {
                projectDecodeRetirementReservation(
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

    private void projectDecodeRetirementReservation(
            DecodeEndpoint retiredEndpoint,
            DecodeEndpoint.ReservationHandle reservation) {
        RequestSlot slot = scheduler.requestSlot(reservation.requestId());
        if (slot == null) {
            return;
        }
        String detail = "Decode endpoint generation retired: generation="
                + reservation.endpointGenerationId();
        RequestRegistry.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)) {
                return;
            }
            work = scheduler.materializePreemptionWorkLocked(
                    slot,
                    slot.reduceDecodeGenerationRetired(
                            retiredEndpoint, reservation, detail),
                    null);
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private static void logEventProjectionFailureNoFail(
            String event,
            Throwable failure) {
        try {
            Logger.error("Endpoint event projection isolated: event={}",
                    event, failure);
        } catch (Throwable ignoredDiagnosticFailure) {
            // Endpoint ownership is already committed; diagnostics are leaves.
        }
    }

    private static void logErrorNoFail(
            String format,
            long first,
            String second,
            Throwable failure) {
        try {
            Logger.error(format, first, second, failure);
        } catch (Throwable ignoredDiagnosticFailure) {
            // Continue projecting the remaining exact facts.
        }
    }

    private static void logErrorNoFail(
            String format,
            long first,
            long second,
            Throwable failure) {
        try {
            Logger.error(format, first, second, failure);
        } catch (Throwable ignoredDiagnosticFailure) {
            // Continue projecting the remaining exact facts.
        }
    }

    private static void logErrorNoFail(
            String format,
            long value,
            Throwable failure) {
        try {
            Logger.error(format, value, failure);
        } catch (Throwable ignoredDiagnosticFailure) {
            // Continue projecting the remaining exact facts.
        }
    }

    private static void logErrorNoFail(
            String format,
            long value) {
        try {
            Logger.error(format, value);
        } catch (Throwable ignoredDiagnosticFailure) {
            // This diagnostic cannot change endpoint-event reduction.
        }
    }

    private static void logWarnNoFail(
            String format,
            long first,
            String second) {
        try {
            Logger.warn(format, first, second);
        } catch (Throwable ignoredDiagnosticFailure) {
            // This diagnostic cannot change endpoint-event reduction.
        }
    }
}
