package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.WorkerBatcher.QueueSnapshot;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/** Router-backed placement for eviction admissions, outside slot lifecycle. */
@Component
public class EvictionPlacement {

    private final DefaultRouter router;
    private final CostBasedPrefillStrategy prefillSelector;
    private final RequestRegistry lifecycle;
    private final BatchSchedulerReporter reporter;

    public EvictionPlacement(
            DefaultRouter router,
            CostBasedPrefillStrategy prefillSelector,
            RequestRegistry lifecycle,
            BatchSchedulerReporter reporter) {
        this.router = Objects.requireNonNull(router, "router");
        this.prefillSelector = Objects.requireNonNull(
                prefillSelector, "prefillSelector");
        this.lifecycle = Objects.requireNonNull(lifecycle, "lifecycle");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
    }

    /** Empty means the exact reservation was published successfully. */
    public Optional<Response> placeReservedDecode(
            BalanceContext context,
            CompletableFuture<Response> future,
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation) {
        if (context.getRequestId() != reservation.requestId()) {
            throw new IllegalArgumentException(
                    "Decode reservation belongs to another request");
        }

        ServerStatus decode = buildReservedDecodeStatus(
                context, endpoint, reservation);
        if (decode == null) {
            return failReservedDecode(
                    endpoint,
                    reservation,
                    "Decode generation retired before canonical placement");
        }

        SelectedRole prefillSelection;
        try {
            prefillSelection = prefillSelector.select(
                    context, RoleType.PREFILL, decode.getGroup());
        } catch (RuntimeException | Error failure) {
            endpoint.releaseReservationExact(reservation);
            throw failure;
        }
        if (prefillSelection == null) {
            return failReservedDecode(
                    endpoint,
                    reservation,
                    "no Prefill worker for reserved Decode placement");
        }

        Response routeResponse = new Response();
        try {
            routeResponse.setSuccess(true);
            routeResponse.setServerStatus(List.of(
                    prefillSelection.serverStatus(), decode));
        } catch (RuntimeException | Error failure) {
            prefillSelection.close();
            endpoint.releaseReservationExact(reservation);
            throw failure;
        }

        QueueRouteAdmission admission;
        try (prefillSelection) {
            admission = QueueRouteAdmission.tryPrepareExistingDecode(
                    prefillSelection,
                    endpoint,
                    reservation,
                    decode,
                    routeResponse);
            if (admission == null) {
                return Optional.of(resourceExhausted(
                        "Decode generation retired before queue admission"));
            }
        }

        try (admission) {
            ScheduledRequest item = admission.buildItem(
                    context, future, System.currentTimeMillis());
            context.setRouteSubmittedNanos(System.nanoTime());
            if (!admission.commitTo(lifecycle, item, true)) {
                return Optional.of(resourceExhausted(
                        "admission capacity is temporarily exhausted"));
            }
            ServerStatus prefill = item.prefill();
            context.setScheduledPrefillEndpoint(
                    prefill.getServerIp() + ":" + prefill.getHttpPort());
            reportPlacement(context, item, "reserved Decode");
            return Optional.empty();
        }
    }

    public PrefillEviction preparePrefillEviction(
            BalanceContext context,
            CompletableFuture<Response> future) {
        PlacementResult<QueueRouteAdmission, PlacementKey> routing =
                router.routeForQueue(context);
        if (routing.status() != PlacementResult.Status.SUCCESS) {
            return null;
        }
        QueueRouteAdmission admission = routing.value();
        try {
            ScheduledRequest item = admission.buildItem(
                    context, future, System.currentTimeMillis());
            DecodeEndpoint decodeEndpoint = item.decodeEp();
            long seqLen = context.getRequest().getSeqLen();
            long maxNewTokens = context.getRequest().getMaxNewTokens();
            long kvTotal = decodeEndpoint == null
                    ? 0L : decodeEndpoint.realKvTotal();
            PriorityRequestEnvelope envelope = new PriorityRequestEnvelope(
                    context.getRequestId(),
                    context.getPriority(),
                    seqLen,
                    maxNewTokens,
                    context.getStartTime(),
                    seqLen,
                    context.getConfig().decodeKvReservationTokens(
                            seqLen, maxNewTokens, kvTotal));
            QueueSnapshot snapshot = item.prefillEp()
                    .captureQueueSnapshot();
            return new PrefillEviction(
                    context, admission, item, envelope, snapshot);
        } catch (RuntimeException | Error failure) {
            admission.close();
            throw failure;
        }
    }

    public final class PrefillEviction implements AutoCloseable {
        private final BalanceContext context;
        private final ScheduledRequest item;
        private final PriorityRequestEnvelope envelope;
        private final QueueSnapshot queueSnapshot;
        private QueueRouteAdmission admission;
        private boolean attempted;

        private PrefillEviction(
                BalanceContext context,
                QueueRouteAdmission admission,
                ScheduledRequest item,
                PriorityRequestEnvelope envelope,
                QueueSnapshot queueSnapshot) {
            this.context = context;
            this.admission = admission;
            this.item = item;
            this.envelope = envelope;
            this.queueSnapshot = queueSnapshot;
        }

        public PriorityRequestEnvelope envelope() {
            return envelope;
        }

        public QueueSnapshot queueSnapshot() {
            return queueSnapshot;
        }

        public WorkerBatcher.QueueReplacementStatus commit(
                List<ScheduledRequest> exactVictims) {
            if (attempted || admission == null) {
                throw new IllegalStateException(
                        "Prefill eviction admission was already consumed");
            }
            attempted = true;
            WorkerBatcher.QueueReplacementStatus status =
                    admission.commitReplacingQueuedVictims(
                            lifecycle, item, true, exactVictims);
            if (status == WorkerBatcher.QueueReplacementStatus.SUCCESS) {
                admission = null;
                publishCommittedPlacementMetadata(context, item);
            }
            return status;
        }

        public void close() {
            QueueRouteAdmission owned = admission;
            admission = null;
            if (owned != null) {
                owned.close();
            }
        }
    }

    /** Optional route metadata cannot hide an already committed replacement. */
    private void publishCommittedPlacementMetadata(
            BalanceContext context,
            ScheduledRequest item) {
        try {
            context.setRouteSubmittedNanos(System.nanoTime());
            ServerStatus prefill = item.prefill();
            context.setScheduledPrefillEndpoint(
                    prefill.getServerIp() + ":" + prefill.getHttpPort());
            reportPlacement(context, item, "Prefill eviction");
        } catch (Throwable metadataFailure) {
            try {
                Logger.warn(
                        "Committed Prefill replacement metadata was isolated: request_id={}",
                        context.getRequestId(),
                        metadataFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Canonical replacement cannot depend on diagnostics.
            }
        }
    }

    private void reportPlacement(
            BalanceContext context,
            ScheduledRequest item,
            String kind) {
        try {
            reporter.reportRouteSubmitTimeMs(
                    RoleType.PREFILL.name(),
                    item.prefillEp().getIp(),
                    System.currentTimeMillis() - context.getStartTime());
        } catch (RuntimeException telemetryFailure) {
            Logger.warn(
                    "Failed to report {} placement: request_id={}",
                    kind,
                    context.getRequestId(),
                    telemetryFailure);
        }
    }

    private static Optional<Response> failReservedDecode(
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation,
            String message) {
        endpoint.releaseReservationExact(reservation);
        return Optional.of(resourceExhausted(message));
    }

    private static Response resourceExhausted(String message) {
        Response response = Response.error(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        response.setErrorMessage(
                StrategyErrorType.RESOURCE_EXHAUSTED.buildErrorMessage(message));
        return response;
    }

    private static ServerStatus buildReservedDecodeStatus(
            BalanceContext context,
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation) {
        WorkerStatus worker = endpoint.getStatus();
        if (reservation.requestId() != context.getRequestId()
                || reservation.endpointGenerationId()
                        != worker.getGenerationId()
                || endpoint.isRetired()) {
            return null;
        }
        WorkerStatus.TopologySnapshot topology = worker.topologySnapshot();
        WorkerStatus.EngineObservation engine =
                worker.committedEngineObservation();
        ServerStatus status = new ServerStatus();
        status.setSuccess(true);
        status.setRole(RoleType.DECODE);
        status.setServerIp(topology.ip());
        status.setHttpPort(topology.port());
        status.setGrpcPort(CommonUtils.toGrpcPort(topology.port()));
        status.setDpRank(engine.dpRank());
        status.setGroup(topology.group());
        status.setRequestId(context.getRequestId());
        return status;
    }

}
