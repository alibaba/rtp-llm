package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.DecodeSelectionStrategy;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.VitSelectionStrategy;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheSelector;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Component
public class RequestWorkerSelector {

    private final PrefillSelectionStrategy prefillSelector;
    private final DecodeSelectionStrategy decodeSelector;
    private final DecodeSelectionStrategy.CapacityReader capacityReader = this::readDecodeCapacity;
    private final VitSelectionStrategy vitSelector;
    private final List<RoleType> requiredRoles;
    private final VitCacheSelector vitCacheSelector;
    private final EndpointRegistry endpoints;
    private final EngineHealthReporter healthReporter;
    private final CacheMetricsReporter cacheReporter;
    private final boolean queueScheduling;
    private final String deliveryMode;
    private static final ThreadLocal<DecodeInputs> DECODE_INPUTS =
            ThreadLocal.withInitial(DecodeInputs::new);

    @Autowired
    public RequestWorkerSelector(
            EndpointRegistry endpoints,
            PrefillSelectionStrategy prefillSelector,
            DecodeSelectionStrategy decodeSelector,
            VitSelectionStrategy vitSelector,
            VitCacheSelector vitCacheSelector,
            EngineHealthReporter healthReporter,
            CacheMetricsReporter cacheReporter,
            ConfigService configService,
            ModelMetaConfig modelMetaConfig) {
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.vitCacheSelector = vitCacheSelector;
        this.healthReporter = Objects.requireNonNull(healthReporter);
        this.cacheReporter = Objects.requireNonNull(cacheReporter);
        this.queueScheduling = configService.loadBalanceConfig().isQueue();
        this.deliveryMode = configService.loadBalanceConfig().getDispatcher().typeName();
        this.prefillSelector = Objects.requireNonNull(prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(vitSelector, "vitSelector");
        this.requiredRoles =
                List.copyOf(
                        Objects.requireNonNull(modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    public PlacementResult<RequestRoute, PlacementKey> select(
            RequestContext context, String policyGroup) {
        Request request = context == null ? null : context.data().getRequest();
        SchedulingMetadata metadata =
                context == null ? null : context.data().getSchedulingMetadata();
        if (request == null || metadata == null) {
            Logger.error("request has no scheduling inputs");
            return PlacementResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        if (StringUtils.isNotBlank(policyGroup)) {
            Logger.info(
                    "Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.data().getRequestId(),
                    "trafficPolicy",
                    policyGroup);
        }
        RequestRoute routing = new RequestRoute(context);
        boolean complete = false;
        Throwable primary = null;
        try {
            String group = policyGroup;
            if (request.getSelectedVit() != null) {
                ServerStatus vit =
                        vitCacheSelector == null || !requiredRoles.contains(RoleType.VIT)
                                ? null
                                : vitCacheSelector.selectPinned(context, policyGroup, routing);
                if (vit == null) {
                    return PlacementResult.rejected(
                            Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                group = vit.getGroup();
            }
            for (RoleType role : requiredRoles) {
                if (role == RoleType.VIT && request.getSelectedVit() != null) {
                    continue;
                }
                PlacementResult<ServerStatus, PlacementKey> result =
                        selectRole(request, metadata, role, group, routing);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    Logger.debug(
                            "Failed to select {} worker for request {}",
                            role.getCode(),
                            context.data().getRequestId());
                    return switch (result.status()) {
                        case REJECTED, BLOCKED -> result.asFailure();
                        default ->
                                throw new IllegalStateException(
                                        "unexpected selector result: " + result.status());
                    };
                }
                ServerStatus selection = result.value();
                if (StringUtils.isBlank(policyGroup)) {
                    group = selection.getGroup();
                }
            }
            var result = PlacementResult.<RequestRoute, PlacementKey>success(routing.prepare());
            complete = true;
            return result;
        } catch (RuntimeException | Error failure) {
            primary = failure;
            throw failure;
        } finally {
            if (!complete) {
                Throwable cleanup = Failures.close(routing);
                if (primary != null) {
                    Failures.append(primary, cleanup);
                } else {
                    Failures.rethrow(cleanup, "route selection cleanup failed");
                }
            }
        }
    }

    public Response routeVit(RequestContext context) {
        if (context == null || context.data().getRequest() == null) {
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        if (!requiredRoles.contains(RoleType.VIT) || vitCacheSelector == null) {
            return Response.error(StrategyErrorType.NO_VIT_WORKER);
        }
        ServerStatus vit = vitCacheSelector.select(context, resolvePolicyGroup(context));
        Response response = new Response();
        response.setSuccess(vit.isSuccess());
        response.setCode(vit.getCode());
        response.setErrorMessage(vit.getMessage());
        if (vit.isSuccess()) {
            response.setServerStatus(List.of(vit));
        }
        return response;
    }

    public boolean selectedVitIsValid(RequestContext context) {
        if (context.data().getRequest().getSelectedVit() == null) {
            return true;
        }
        return vitCacheSelector != null
                && requiredRoles.contains(RoleType.VIT)
                && vitCacheSelector.validate(context, resolvePolicyGroup(context)).isSuccess();
    }

    String resolvePolicyGroup(RequestContext context) {
        if (context == null || context.data().getRequest() == null) {
            return null;
        }
        var selector = context.data().getConfig().getRouter().getGroupSelector();
        Request request = context.data().getRequest();
        return selector == null
                ? null
                : selector.resolveTargetGroup(
                                request.getRequestId(), request.getApiKey(), request.getSeqLen())
                        .orElse(null);
    }

    private PlacementResult<ServerStatus, PlacementKey> selectRole(
            Request request,
            SchedulingMetadata metadata,
            RoleType role,
            String group,
            RequestRoute routing) {
        return switch (role) {
            case PREFILL, PDFUSION -> selectPrefill(request, metadata, role, group, routing);
            case DECODE -> selectDecode(request, metadata, group, routing);
            case VIT -> {
                var selected = selectVit(request.getRequestId(), group, routing);
                yield selected == null
                        ? PlacementResult.blocked(new PlacementKey(role, group, null))
                        : PlacementResult.success(selected);
            }
            case FRONTEND ->
                    throw new IllegalArgumentException(
                            "Endpoint selection is not supported for FRONTEND");
        };
    }

    private ServerStatus selectVit(long requestId, String group, RequestRoute route) {
        List<String> choices = endpoints.endpointAddressSnapshot(RoleType.VIT);
        ArrayList<String> remaining = null;
        while (!choices.isEmpty()) {
            int index = vitSelector.selectIndex(choices);
            com.google.common.base.Preconditions.checkArgument(
                    index >= 0 && index < choices.size(), "VIT strategy returned an invalid index");
            var selected = captureVit(choices.get(index), group, requestId, route);
            if (selected != null) return selected;
            if (remaining == null) {
                remaining = new ArrayList<>(choices);
                choices = java.util.Collections.unmodifiableList(remaining);
            }
            java.util.Collections.swap(remaining, index, remaining.size() - 1);
            remaining.removeLast();
        }
        return null;
    }

    private ServerStatus captureVit(
            String address, String group, long requestId, RequestRoute route) {
        WorkerEndpoint.GenerationPin pin = endpoints.capture(RoleType.VIT, address);
        if (pin == null) {
            return null;
        }
        try {
            var status = pin.endpoint().getStatus();
            var topology = status.topologySnapshot();
            if (group != null && !group.equals(topology.group())) {
                return null;
            }
            var engine = status.committedEngineObservation();
            ServerStatus result = ServerStatus.forWorker(RoleType.VIT, requestId, topology, engine);

            WorkerEndpoint.GenerationPin selectedPin = pin;
            pin = null;
            return route.add(selectedPin, result, -1L, 0L);
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }

    private PlacementResult<ServerStatus, PlacementKey> selectPrefill(
            Request request,
            SchedulingMetadata metadata,
            RoleType role,
            String group,
            RequestRoute route) {
        var directory = endpoints.prefillRoutingSnapshot(role);
        var result =
                prefillSelector.select(
                        request, metadata, role, group, directory, System::currentTimeMillis);
        if (result.status() != PlacementResult.Status.SUCCESS) {
            if (result.diagnostics() != null
                    && result.diagnostics().get("details") instanceof Map<?, ?> details
                    && details.get("affinityReason") instanceof String reason
                    && !reason.isEmpty()) {
                for (var input : directory) {
                    if (input.address().equals(details.get("selectedEndpoint"))) {
                        reportCacheAffinity(role, input.endpoint().getIp(), reason);
                        break;
                    }
                }
            }
            return result.asFailure();
        }
        var selected = result.value();
        PrefillEndpoint endpoint = selected.input().endpoint();
        reportCacheAffinity(role, endpoint.getIp(), selected.affinityReason());
        WorkerEndpoint.GenerationPin pin = endpoints.capture(role, selected.address());
        try {
            if (pin == null
                    || pin.endpoint() != endpoint
                    || pin.generationId() != selected.generationId()) {
                return PlacementResult.blocked(new PlacementKey(role, group, null));
            }
            var failure = prefillCapacityFailure(endpoint, metadata, role, selected);
            if (failure != null) {
                return failure;
            }
            var status = endpoint.getStatus();
            ServerStatus assignment =
                    ServerStatus.forWorker(
                            role,
                            request.getRequestId(),
                            status.topologySnapshot(),
                            status.committedEngineObservation());
            if (selected.projectedTtftMs() >= 0L) {
                assignment.setPrefillTime(selected.projectedTtftMs());
            }
            DebugInfo debug = new DebugInfo();
            debug.setHitCacheLen(selected.cacheHitTokens());
            assignment.setDebugInfo(debug);
            var ownedPin = pin;
            pin = null;
            route.add(ownedPin, assignment, selected.prefillWorkMs(), selected.ownershipVersion());
            reportSelection(role, endpoint.getIp(), request.getSeqLen(), selected);
            return PlacementResult.success(assignment);
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }

    private void reportCacheAffinity(RoleType role, String ip, String reason) {
        if (reason == null) return;
        try {
            cacheReporter.reportCacheAffinityDecision(role, ip, reason);
        } catch (Throwable telemetryFailure) {
            Logger.warn("Cache affinity telemetry failed", telemetryFailure);
        }
    }

    private PlacementResult<ServerStatus, PlacementKey> prefillCapacityFailure(
            PrefillEndpoint endpoint,
            SchedulingMetadata metadata,
            RoleType role,
            PrefillSelectionStrategy.Selection<?> selected) {
        if (queueScheduling
                || endpoint.ownership.requestCapacityAvailable(endpoint.requestLimit, 1)) {
            return null;
        }
        Response failure = Response.error(StrategyErrorType.RESOURCE_EXHAUSTED);
        var reason =
                classifyWorkerCapacity(
                        endpoint.ownership.admissionSummary(
                                metadata.priority(), endpoint.requestLimit));
        failure =
                reason == AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                                || reason == AdmissionRejectReason.SAME_PRIORITY_AHEAD
                        ? Response.error(StrategyErrorType.PRIORITY_ADMISSION_REJECTED, reason)
                        : reason == AdmissionRejectReason.UNSPECIFIED
                                ? Response.error(StrategyErrorType.ADMISSION_UNAVAILABLE)
                                : failure;
        return PlacementResult.blocked(
                PlacementKey.exact(
                        role, endpoint.getStatus().topologySnapshot().group(), selected.address()),
                failure,
                Map.of(
                        "role",
                        role.name(),
                        "workers",
                        selected.workerCount(),
                        "observedAtMs",
                        System.currentTimeMillis(),
                        "details",
                        Map.of(
                                "reason",
                                "PREFILL_CAPACITY",
                                "selectedEndpoint",
                                selected.address(),
                                "cacheHitTokens",
                                selected.cacheHitTokens())));
    }

    private static AdmissionRejectReason classifyWorkerCapacity(
            PrefillState.AdmissionSummary summary) {
        long residual = Math.max(0L, summary.requestsToRelease() - summary.lowerPriorityRequests());
        long protectedRequests = summary.higherPriorityRequests() + summary.samePriorityRequests();
        if (residual > protectedRequests && summary.unknownPriorityRequests() > 0L) {
            return AdmissionRejectReason.UNSPECIFIED;
        }
        if (residual > 0L && protectedRequests >= residual) {
            return summary.higherPriorityRequests() > 0L
                    ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                    : AdmissionRejectReason.SAME_PRIORITY_AHEAD;
        }
        return AdmissionRejectReason.RESOURCE_EXHAUSTED;
    }

    private void reportSelection(
            RoleType role, String ip, long seqLen, PrefillSelectionStrategy.Selection<?> selected) {
        try {
            if (selected.projectedTtftMs() >= 0L) {
                healthReporter.reportPrefillSelectedEstimates(
                        role,
                        ip,
                        deliveryMode,
                        selected.projectedTtftMs(),
                        selected.prefillWorkMs());
            }
        } catch (Throwable telemetryFailure) {
            Logger.warn("Prefill selected-estimate telemetry failed", telemetryFailure);
        }
        try {
            cacheReporter.reportSelectionCache(
                    role,
                    seqLen,
                    selected.cacheHitTokens(),
                    selected.routingCacheMatchTokens(),
                    selected.maximumRoutingCacheMatchTokens());
        } catch (Throwable telemetryFailure) {
            Logger.warn("Prefill cache telemetry failed", telemetryFailure);
        }
    }

    private PlacementResult<ServerStatus, PlacementKey> selectDecode(
            Request request, SchedulingMetadata metadata, String group, RequestRoute route) {
        DecodeInputs cached = DECODE_INPUTS.get();
        DecodeInputs inputs = cached.inUse ? new DecodeInputs() : cached;
        inputs.inUse = true;
        int captured = 0;
        try {
            for (int attempt = 0; attempt < 2; attempt++) {
                inputs.decode.clear();
                endpoints.decodeRoutingSnapshot(group, inputs.decode);
                captured = Math.max(captured, inputs.decode.size());
                var result =
                        decodeSelector.select(
                                request, metadata, group, inputs.readOnly, capacityReader);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    return result.asFailure();
                }
                var selected = result.value();
                WorkerEndpoint.GenerationPin pin = endpoints.captureDecodeGeneration(selected);
                if (pin == null) {
                    continue;
                }
                try {
                    com.google.common.base.Preconditions.checkState(
                            pin.generationId() == selected.generationId()
                                    && pin.endpoint() instanceof DecodeEndpoint,
                            "Decode snapshot pin changed before selection handoff");
                    ServerStatus assignment =
                            ServerStatus.forWorker(
                                    RoleType.DECODE,
                                    request.getRequestId(),
                                    selected.topology(),
                                    selected.workerStatus().fields());
                    var ownedPin = pin;
                    pin = null;
                    return PlacementResult.success(
                            route.add(ownedPin, assignment, -1L, selected.admissionVersion()));
                } finally {
                    if (pin != null) {
                        pin.close();
                    }
                }
            }
            return PlacementResult.blocked(new PlacementKey(RoleType.DECODE, group, null));
        } finally {
            captured = Math.max(captured, inputs.decode.size());
            inputs.decode.clear();
            inputs.inUse = false;
            if (inputs == cached && captured > 4096) {
                DECODE_INPUTS.remove();
            }
        }
    }

    private org.flexlb.balance.endpoint.DecodeResources.AdmissionSummary readDecodeCapacity(
            DecodeRoutingView view) {
        try (var pin = endpoints.captureDecodeGeneration(view)) {
            return pin == null ? null : ((DecodeEndpoint) pin.endpoint()).admissionSummary();
        }
    }

    private static final class DecodeInputs {
        final ArrayList<DecodeRoutingView> decode = new ArrayList<>();
        final List<DecodeRoutingView> readOnly = java.util.Collections.unmodifiableList(decode);
        boolean inUse;
    }
}
