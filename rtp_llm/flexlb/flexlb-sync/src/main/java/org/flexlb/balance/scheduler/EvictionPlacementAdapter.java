package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointGenerationRetiredException;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime.QueueSnapshot;
import org.flexlb.balance.eviction.EvictionPlacementPort;
import org.flexlb.balance.eviction.model.PriorityRequestEnvelope;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
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
import java.util.concurrent.CompletableFuture;

/** Router-backed placement for eviction admissions, outside slot lifecycle. */
@Component
final class EvictionPlacementAdapter implements EvictionPlacementPort {

    private final Router router;
    private final ConfiguredLoadBalanceSelector endpointSelector;
    private final InflightCommitPort inflight;
    private final BatchSchedulerReporter reporter;

    EvictionPlacementAdapter(
            Router router,
            ConfiguredLoadBalanceSelector endpointSelector,
            InflightCommitPort inflight,
            BatchSchedulerReporter reporter) {
        this.router = Objects.requireNonNull(router, "router");
        this.endpointSelector = Objects.requireNonNull(
                endpointSelector, "endpointSelector");
        this.inflight = Objects.requireNonNull(inflight, "inflight");
        this.reporter = Objects.requireNonNull(reporter, "reporter");
    }

    @Override
    public DecodePlacement placeReservedDecode(
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
            prefillSelection = endpointSelector.select(
                    context, RoleType.PREFILL, decode.getGroup());
        } catch (RuntimeException | Error failure) {
            endpoint.rollbackExact(reservation);
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
            endpoint.rollbackExact(reservation);
            throw failure;
        }

        QueueRouteAdmission admission;
        try (prefillSelection) {
            admission = QueueRouteAdmission.prepareExistingDecode(
                    prefillSelection,
                    endpoint,
                    reservation,
                    decode,
                    routeResponse);
        } catch (EndpointGenerationRetiredException retired) {
            return new DecodePlacement.Failed(new AdmissionFailure(
                    StrategyErrorType.RESOURCE_EXHAUSTED,
                    AdmissionRejectReason.RESOURCE_EXHAUSTED,
                    retired.getMessage()));
        }

        try (admission) {
            BatchItem item = admission.buildItem(
                    context, future, System.currentTimeMillis());
            context.setRouteSubmittedNanos(System.nanoTime());
            if (!admission.commitTo(inflight, item, true)) {
                return new DecodePlacement.Failed(
                        AdmissionFailure.resourceExhausted());
            }
            ServerStatus prefill = item.prefill();
            context.setScheduledPrefillEndpoint(
                    prefill.getServerIp() + ":" + prefill.getHttpPort());
            reportPlacement(context, item, "reserved Decode");
            return new DecodePlacement.Committed();
        }
    }

    @Override
    public PrefillEvictionAdmission preparePrefillEviction(
            BalanceContext context,
            CompletableFuture<Response> future) {
        QueueRoutingResult routing = router.routeForQueue(context);
        if (!(routing instanceof QueueRoutingResult.Admitted admitted)) {
            return null;
        }
        QueueRouteAdmission admission = admitted.admission();
        try {
            BatchItem item = admission.buildItem(
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
            return new PreparedPrefillEviction(
                    context, admission, item, envelope, snapshot);
        } catch (RuntimeException | Error failure) {
            admission.close();
            throw failure;
        }
    }

    private final class PreparedPrefillEviction
            implements PrefillEvictionAdmission {
        private final BalanceContext context;
        private final BatchItem item;
        private final PriorityRequestEnvelope envelope;
        private final QueueSnapshot queueSnapshot;
        private QueueRouteAdmission admission;
        private boolean attempted;

        private PreparedPrefillEviction(
                BalanceContext context,
                QueueRouteAdmission admission,
                BatchItem item,
                PriorityRequestEnvelope envelope,
                QueueSnapshot queueSnapshot) {
            this.context = context;
            this.admission = admission;
            this.item = item;
            this.envelope = envelope;
            this.queueSnapshot = queueSnapshot;
        }

        @Override
        public PriorityRequestEnvelope envelope() {
            return envelope;
        }

        @Override
        public QueueSnapshot queueSnapshot() {
            return queueSnapshot;
        }

        @Override
        public PrefillEvictionCommit commit(
                List<DeliveryItem> exactVictims) {
            if (attempted || admission == null) {
                throw new IllegalStateException(
                        "Prefill eviction admission was already consumed");
            }
            attempted = true;
            PreparedEvictionCommits prepared =
                    PreparedEvictionCommits.forVictims(exactVictims);
            QueueRouteAdmission.ReplacementCommit result =
                    admission.commitReplacingQueuedVictims(
                            inflight, item, true, exactVictims);
            PrefillEvictionCommit commit = prepared.resolve(result.status());
            if (commit.status() == PrefillEvictionStatus.COMMITTED) {
                admission = null;
                publishCommittedPlacementMetadata(context, item);
            }
            return commit;
        }

        @Override
        public void close() {
            QueueRouteAdmission owned = admission;
            admission = null;
            if (owned != null) {
                owned.close();
            }
        }
    }

    /** All port results exist before the exact queue replacement can commit. */
    private record PreparedEvictionCommits(
            PrefillEvictionCommit committed,
            PrefillEvictionCommit conflict,
            PrefillEvictionCommit declined) {

        private static PreparedEvictionCommits forVictims(
                List<DeliveryItem> exactVictims) {
            return new PreparedEvictionCommits(
                    new PrefillEvictionCommit(
                            PrefillEvictionStatus.COMMITTED, exactVictims),
                    new PrefillEvictionCommit(
                            PrefillEvictionStatus.CONFLICT, List.of()),
                    new PrefillEvictionCommit(
                            PrefillEvictionStatus.DECLINED, List.of()));
        }

        private PrefillEvictionCommit resolve(
                QueueRouteAdmission.ReplacementStatus status) {
            return switch (status) {
                case SUCCESS -> committed;
                case CONFLICT, NOT_ATTEMPTED -> conflict;
                case DECLINED -> declined;
            };
        }
    }

    /** Optional route metadata cannot hide an already committed replacement. */
    private void publishCommittedPlacementMetadata(
            BalanceContext context,
            BatchItem item) {
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
            BatchItem item,
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

    private static DecodePlacement failReservedDecode(
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation,
            String message) {
        endpoint.rollbackExact(reservation);
        return new DecodePlacement.Failed(new AdmissionFailure(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED,
                message));
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
