package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointEvent;
import org.flexlb.balance.endpoint.EndpointStatusReduction;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.util.Logger;

import java.util.List;
import java.util.Objects;

/**
 * Total projection boundary from endpoint-owned facts into exact RequestSlot
 * transitions. Endpoint accounting is already committed before this class is
 * called; stale generations are legal no-ops and are never queued for replay.
 */
final class EndpointEventProjector {
    private final RequestLifecycleCoordinator scheduler;

    EndpointEventProjector(RequestLifecycleCoordinator scheduler) {
        this.scheduler = Objects.requireNonNull(scheduler, "scheduler");
    }

    void project(EndpointEvent event) {
        if (event == null) {
            return;
        }
        try {
            switch (event) {
                case EndpointEvent.StatusReduced status ->
                        projectStatus(status.reduction());
                case EndpointEvent.PrefillGenerationRetired retired ->
                        projectPrefillRetirement(
                                retired.endpoint(), retired.ownedItems());
                case EndpointEvent.DecodeGenerationRetired retired ->
                        projectDecodeRetirement(
                                retired.endpoint(), retired.ownedReservations());
            }
        } catch (Throwable failure) {
            logEventProjectionFailureNoFail(event, failure);
        }
    }

    private void projectStatus(EndpointStatusReduction reduction) {
        switch (reduction) {
            case EndpointStatusReduction.None ignored -> { }
            case PrefillEndpoint.StatusReduction prefill ->
                    projectPrefillStatus(prefill);
            case DecodeEndpoint.StatusReduction decode ->
                    projectDecodeStatus(decode);
        }
    }

    private void projectPrefillStatus(
            PrefillEndpoint.StatusReduction reduction) {
        for (PrefillWorkLedger.WorkerStatusFact fact : reduction.facts()) {
            try {
                switch (fact) {
                    case PrefillWorkLedger.ActiveWorkerStatusFact active ->
                            projectPrefillActive(reduction.source(), active);
                    case PrefillWorkLedger.TerminalWorkerStatusFact terminal ->
                            projectPrefillTerminal(reduction, terminal);
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
            PrefillWorkLedger.ActiveWorkerStatusFact fact) {
        BatchItem item = (BatchItem) fact.item();
        RequestSlot slot = scheduler.requestSlot(item.requestId());
        if (slot == null) {
            return;
        }
        RequestLifecycleCoordinator.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)) {
                return;
            }
            work = scheduler.reducePreemptionFactLocked(
                    slot,
                    new RequestSlot.PreemptionFact.PrefillActive(
                            source, item),
                    null);
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private void projectPrefillTerminal(
            PrefillEndpoint.StatusReduction reduction,
            PrefillWorkLedger.TerminalWorkerStatusFact fact) {
        if (fact.kind() == PrefillWorkLedger.TerminalFactKind.COMPLETED
                && reduction.semantics()
                    != PrefillEndpoint.StatusSemantics.FUSION_TERMINAL) {
            logWarnNoFail(
                    "Ignoring Prefill-stage successful terminal projection: request_id={} engine={}",
                    fact.item().requestId(), reduction.source().getIp());
            return;
        }

        BatchItem item = (BatchItem) fact.item();
        RequestSlot slot = scheduler.requestSlot(item.requestId());
        if (slot == null) {
            return;
        }
        RequestLifecycleCoordinator.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)
                    || !slot.ownsPrefillFact(reduction.source(), item)) {
                return;
            }
            WorkerTerminalObservation observation =
                    new WorkerTerminalObservation(
                            WorkerTerminalSource.PREFILL_BACKED,
                            fact.kind()
                                == PrefillWorkLedger.TerminalFactKind.COMPLETED,
                            fact.errorCode());
            DeferredTerminal terminal = DeferredTerminal.worker(observation);
            if (fact.kind()
                    == PrefillWorkLedger.TerminalFactKind.PRIORITY_CANCELED) {
                work = scheduler.reducePreemptionFactLocked(
                        slot,
                        new RequestSlot.PreemptionFact.PriorityCanceled(
                                reduction.source(), item, terminal),
                        null);
            } else {
                work = scheduler.reducePreemptionFactLocked(
                        slot,
                        new RequestSlot.PreemptionFact.WorkerTerminal(
                                item, terminal),
                        null);
            }
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private void projectDecodeStatus(DecodeEndpoint.StatusReduction reduction) {
        for (DecodeEndpoint.WorkerStatusFact fact : reduction.facts()) {
            try {
                switch (fact) {
                    case DecodeEndpoint.AcceptedWorkerStatusFact accepted ->
                            projectDecodeAccepted(reduction.source(), accepted);
                    case DecodeEndpoint.TerminalWorkerStatusFact terminal ->
                            projectDecodeTerminal(reduction.source(), terminal);
                }
            } catch (Throwable failure) {
                logErrorNoFail(
                        "Decode status fact projection isolated: request_id={} engine={}",
                        fact.reservation().requestId(),
                        reduction.source().getIp(), failure);
            }
        }
    }

    private void projectDecodeAccepted(
            DecodeEndpoint source,
            DecodeEndpoint.AcceptedWorkerStatusFact fact) {
        RequestSlot slot = scheduler.requestSlot(fact.reservation().requestId());
        if (slot == null) {
            return;
        }
        DecodeAcceptance acceptance;
        RequestLifecycleCoordinator.PreemptionWork work = null;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)
                    || !slot.ownsDecodeFact(source, fact.reservation())) {
                return;
            }
            acceptance = slot.markDecodeAccepted();
            if (acceptance.acceptedBeforeCancel()
                    && acceptance.releasableFence() != null) {
                work = scheduler.reducePreemptionFactLocked(
                        slot,
                        new RequestSlot.PreemptionFact.DeliveryConfirmed(
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
            failure = RequestLifecycleCoordinator.appendFailure(
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
            DecodeEndpoint.TerminalWorkerStatusFact fact) {
        RequestSlot slot = scheduler.requestSlot(fact.reservation().requestId());
        if (slot == null) {
            return;
        }
        RequestLifecycleCoordinator.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)
                    || !slot.ownsDecodeFact(source, fact.reservation())) {
                return;
            }
            slot.markDecodeTerminalOwned();
            WorkerTerminalObservation observation =
                    new WorkerTerminalObservation(
                            WorkerTerminalSource.DECODE,
                            fact.errorCode() == 0L,
                            fact.errorCode());
            DeferredTerminal terminal = DeferredTerminal.worker(observation);
            work = scheduler.reducePreemptionFactLocked(
                    slot,
                    new RequestSlot.PreemptionFact.WorkerTerminal(
                            slot.activeItem(), terminal),
                    null);
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private void projectPrefillRetirement(
            PrefillEndpoint retiredEndpoint,
            List<DeliveryItem> ownedItems) {
        for (int index = 0; index < ownedItems.size(); index++) {
            DeliveryItem exactItem = ownedItems.get(index);
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
            DeliveryItem exactItem) {
        if (!(exactItem instanceof BatchItem item)
                || item.prefillEp() != retiredEndpoint) {
            logErrorNoFail(
                    "Ignoring Prefill retirement item from another generation: request_id={}",
                    exactItem == null ? -1 : exactItem.requestId());
            return;
        }
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
                    RequestLifecycleCoordinator.buildErrorResponse(
                            StrategyErrorType.BATCH_DISPATCH_FAILED, detail));
        }
        scheduler.submitTerminal(action);
    }

    private void projectDecodeRetirement(
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
        RequestLifecycleCoordinator.PreemptionWork work;
        synchronized (slot) {
            if (!scheduler.isCurrentSlot(slot)) {
                return;
            }
            work = scheduler.reducePreemptionFactLocked(
                    slot,
                    new RequestSlot.PreemptionFact.DecodeGenerationRetired(
                            retiredEndpoint, reservation, detail),
                    null);
        }
        scheduler.consumePreemptionWork(slot, work);
    }

    private static void logEventProjectionFailureNoFail(
            EndpointEvent event,
            Throwable failure) {
        try {
            Logger.error("Endpoint event projection isolated: event={}",
                    event.getClass().getSimpleName(), failure);
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
