package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFailure;
import org.flexlb.balance.admission.AdmissionMutation;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointGenerationRetiredException;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime.PreparedOffer;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime.QueueSnapshot;
import org.flexlb.balance.endpoint.WorkerEndpoint;
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
    public PreparedDecodePlacement prepareDecodePlacement(
            BalanceContext context,
            CompletableFuture<Response> future,
            DecodeEndpoint endpoint) {
        DecodeEndpoint.PlacementHandoff decodeHandoff =
                endpoint.tryAcquirePlacementHandoff();
        if (decodeHandoff == null) {
            return null;
        }

        WorkerEndpoint.GenerationPin prefillPin = null;
        PreparedOffer preparedOffer = null;
        try {
            WorkerStatus.TopologySnapshot decodeTopology =
                    endpoint.getStatus().topologySnapshot();
            SelectedRole prefillSelection = endpointSelector.select(
                    context, RoleType.PREFILL, decodeTopology.group());
            if (prefillSelection == null) {
                return null;
            }
            try (prefillSelection) {
                ServerStatus prefillStatus = RequestLifecycleCoordinator.copyOf(
                        prefillSelection.serverStatus());
                if (prefillStatus.getRequestId() != context.getRequestId()) {
                    throw new IllegalStateException(
                            "Prefill selection belongs to another request");
                }
                prefillPin = prefillSelection.takeGenerationPin();
                if (!(prefillPin.endpoint() instanceof PrefillEndpoint prefill)) {
                    throw new IllegalStateException(
                            "Decode eviction requires a Prefill endpoint");
                }
                try {
                    preparedOffer = prefill.prepareOfferPinned(
                            prefillPin,
                            context.getRequestId(),
                            context.getPriority());
                } catch (EndpointGenerationRetiredException retired) {
                    return null;
                }
                if (preparedOffer == null) {
                    return null;
                }
                PreparedDecodePlacement prepared =
                        new PreparedDecodePlacementImpl(
                                context,
                                future,
                                endpoint,
                                decodeHandoff,
                                prefill,
                                prefillStatus,
                                preparedOffer);
                decodeHandoff = null;
                preparedOffer = null;
                return prepared;
            }
        } finally {
            try {
                if (preparedOffer != null) {
                    preparedOffer.close();
                }
            } finally {
                try {
                    if (prefillPin != null) {
                        prefillPin.close();
                    }
                } finally {
                    if (decodeHandoff != null) {
                        decodeHandoff.close();
                    }
                }
            }
        }
    }

    private final class PreparedDecodePlacementImpl
            implements PreparedDecodePlacement {
        private final BalanceContext context;
        private final CompletableFuture<Response> future;
        private final DecodeEndpoint decodeEndpoint;
        private final PrefillEndpoint prefillEndpoint;
        private final ServerStatus prefillStatus;
        private DecodeEndpoint.PlacementHandoff decodeHandoff;
        private PreparedOffer preparedOffer;

        private PreparedDecodePlacementImpl(
                BalanceContext context,
                CompletableFuture<Response> future,
                DecodeEndpoint decodeEndpoint,
                DecodeEndpoint.PlacementHandoff decodeHandoff,
                PrefillEndpoint prefillEndpoint,
                ServerStatus prefillStatus,
                PreparedOffer preparedOffer) {
            this.context = context;
            this.future = future;
            this.decodeEndpoint = decodeEndpoint;
            this.decodeHandoff = decodeHandoff;
            this.prefillEndpoint = prefillEndpoint;
            this.prefillStatus = prefillStatus;
            this.preparedOffer = preparedOffer;
        }

        @Override
        public boolean seal() {
            return requireOpen().seal();
        }

        @Override
        public DecodePlacement commit(
                DecodeEndpoint.ReservationHandle reservation,
                AdmissionMutation exactMutation) {
            PreparedOffer offer = requireOpen();
            DecodeEndpoint.PlacementHandoff handoff = decodeHandoff;
            boolean committed = false;
            try {
                if (reservation == null
                        || reservation.requestId() != context.getRequestId()
                        || reservation.endpointGenerationId()
                                != handoff.generationId()) {
                    return failed(
                            "Decode reservation does not belong to prepared placement");
                }
                ServerStatus decode = buildReservedDecodeStatus(
                        context, decodeEndpoint, handoff, reservation);
                if (decode == null
                        || decodeEndpoint.markQueuedExact(handoff, reservation)
                                == DecodeEndpoint.MarkQueuedResult.NOT_OWNED) {
                    return failed(
                            "Decode reservation ownership changed before placement");
                }

                Response routeResponse = new Response();
                routeResponse.setSuccess(true);
                routeResponse.setServerStatus(List.of(prefillStatus, decode));
                BatchItem item = new BatchItem(
                        context,
                        future,
                        routeResponse,
                        RequestLifecycleCoordinator.copyOf(prefillStatus),
                        decode,
                        prefillEndpoint,
                        decodeEndpoint,
                        reservation,
                        System.currentTimeMillis());
                if (!inflight.commitInflight(
                        item,
                        true,
                        exactMutation,
                        () -> {
                            offer.commit(item);
                            return true;
                        })) {
                    return new DecodePlacement.Failed(
                            unavailable(
                                    "canonical request ownership changed before placement"));
                }

                committed = true;
                preparedOffer = null;
                decodeHandoff = null;
                handoff.close();
                publishCommittedPlacementMetadata(
                        context, item, "reserved Decode");
                return new DecodePlacement.Committed();
            } catch (RuntimeException placementFailure) {
                Logger.warn(
                        "Prepared Decode placement failed: request_id={}",
                        context.getRequestId(), placementFailure);
                return failed(
                        "canonical Decode placement failed: "
                                + placementFailure.getMessage());
            } finally {
                if (!committed) {
                    if (reservation != null) {
                        decodeEndpoint.rollbackExact(reservation);
                    }
                    close();
                }
            }
        }

        private PreparedOffer requireOpen() {
            PreparedOffer offer = preparedOffer;
            if (offer == null) {
                throw new IllegalStateException(
                        "prepared Decode placement was already consumed");
            }
            if (decodeHandoff == null) {
                throw new IllegalStateException(
                        "prepared Decode generation handoff was already consumed");
            }
            return offer;
        }

        @Override
        public void close() {
            PreparedOffer offer = preparedOffer;
            preparedOffer = null;
            try {
                if (offer != null) {
                    offer.close();
                }
            } finally {
                DecodeEndpoint.PlacementHandoff handoff = decodeHandoff;
                decodeHandoff = null;
                if (handoff != null) {
                    handoff.close();
                }
            }
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
                List<DeliveryItem> exactVictims,
                AdmissionMutation exactMutation) {
            if (attempted || admission == null) {
                throw new IllegalStateException(
                        "Prefill eviction admission was already consumed");
            }
            attempted = true;
            PreparedEvictionCommits prepared =
                    PreparedEvictionCommits.forVictims(exactVictims);
            QueueRouteAdmission.ReplacementCommit result =
                    admission.commitReplacingQueuedVictims(
                            inflight,
                            item,
                            true,
                            exactMutation,
                            exactVictims);
            PrefillEvictionCommit commit = prepared.resolve(result.status());
            if (commit.status() == PrefillEvictionStatus.COMMITTED) {
                admission = null;
                publishCommittedPlacementMetadata(
                        context, item, "Prefill eviction");
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
            BatchItem item,
            String kind) {
        try {
            context.setRouteSubmittedNanos(System.nanoTime());
            ServerStatus prefill = item.prefill();
            context.setScheduledPrefillEndpoint(
                    prefill.getServerIp() + ":" + prefill.getHttpPort());
            reportPlacement(context, item, kind);
        } catch (Throwable metadataFailure) {
            try {
                Logger.warn(
                        "Committed {} metadata was isolated: request_id={}",
                        kind,
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

    private static ServerStatus buildReservedDecodeStatus(
            BalanceContext context,
            DecodeEndpoint endpoint,
            DecodeEndpoint.PlacementHandoff handoff,
            DecodeEndpoint.ReservationHandle reservation) {
        WorkerStatus worker = endpoint.getStatus();
        if (reservation.requestId() != context.getRequestId()
                || reservation.endpointGenerationId()
                        != worker.getGenerationId()
                || reservation.endpointGenerationId()
                        != handoff.generationId()) {
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

    private static DecodePlacement failed(String message) {
        return new DecodePlacement.Failed(unavailable(message));
    }

    private static AdmissionFailure unavailable(String message) {
        return new AdmissionFailure(
                StrategyErrorType.ADMISSION_UNAVAILABLE,
                AdmissionRejectReason.UNSPECIFIED,
                message);
    }
}
