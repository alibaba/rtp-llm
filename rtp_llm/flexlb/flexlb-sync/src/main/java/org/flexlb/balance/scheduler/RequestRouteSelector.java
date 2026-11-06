package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.DecodeSelectionStrategy;
import org.flexlb.balance.strategy.PrefillSelectionStrategy;
import org.flexlb.balance.strategy.VitSelectionStrategy;
import org.flexlb.cache.telemetry.CacheMetricsReporter;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheSelector;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

@Component
public class RequestRouteSelector {

    private final PrefillSelectionStrategy prefillSelector;
    private final DecodeSelectionStrategy decodeSelector;
    private final DecodeSelectionStrategy.CapacityReader capacityReader = this::readDecodeCapacity;
    private final VitSelectionStrategy vitSelector;
    private final org.flexlb.balance.strategy.EncoderSelectionStrategy encoderSelector;
    private final org.flexlb.cache.match.CacheAwareService cacheService;
    private final List<RoleType> requiredRoles;
    private final VitCacheSelector vitCacheSelector;
    private final EndpointRegistry endpoints;
    private final EngineHealthReporter healthReporter;
    private final CacheMetricsReporter cacheReporter;
    private final String deliveryMode;
    private static final ThreadLocal<DecodeInputs> DECODE_INPUTS =
            ThreadLocal.withInitial(DecodeInputs::new);

    @Autowired
    public RequestRouteSelector(
            EndpointRegistry endpoints,
            PrefillSelectionStrategy prefillSelector,
            DecodeSelectionStrategy decodeSelector,
            VitSelectionStrategy vitSelector,
            org.flexlb.balance.strategy.EncoderSelectionStrategy encoderSelector,
            org.flexlb.cache.match.CacheAwareService cacheService,
            VitCacheSelector vitCacheSelector,
            EngineHealthReporter healthReporter,
            CacheMetricsReporter cacheReporter,
            ConfigService configService,
            ModelMetaConfig modelMetaConfig) {
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.encoderSelector = Objects.requireNonNull(encoderSelector);
        this.cacheService = Objects.requireNonNull(cacheService);
        this.vitCacheSelector = vitCacheSelector;
        this.healthReporter = Objects.requireNonNull(healthReporter);
        this.cacheReporter = Objects.requireNonNull(cacheReporter);
        this.deliveryMode = configService.loadBalanceConfig().getDispatcher().typeName();
        this.prefillSelector = Objects.requireNonNull(prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(vitSelector, "vitSelector");
        this.requiredRoles =
                List.copyOf(
                        Objects.requireNonNull(modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    public PlacementResult<RouteResult, PlacementKey> select(
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
        var routing = new RouteResult.Builder(context);
        String group = policyGroup;
        if (request.getSelectedVit() != null) {
            ServerStatus vit =
                    vitCacheSelector == null || !requiredRoles.contains(RoleType.VIT)
                            ? null
                            : vitCacheSelector.selectForRoute(context, policyGroup, routing);
            if (vit == null) {
                return PlacementResult.rejected(Response.error(StrategyErrorType.VIT_ROUTE_STALE));
            }
            group = vit.getGroup();
        }
        var requested = context.data().getRequestedRoles();
        if ((!requested.isEmpty() && !requiredRoles.containsAll(requested))
                || context.data().getPhase() == org.flexlb.dao.route.RequestPhase.ENCODER
                        && !requiredRoles.contains(RoleType.ENCODER)) {
            return PlacementResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        var roles =
                requested.isEmpty()
                        ? (context.data().getPhase() == org.flexlb.dao.route.RequestPhase.ENCODER
                                ? List.of(RoleType.ENCODER)
                                : requiredRoles)
                        : requiredRoles.stream().filter(requested::contains).toList();
        for (RoleType role : roles) {
            if (role == RoleType.VIT && request.getSelectedVit() != null) {
                continue;
            }
            context.data().beginRoutingAttempt(role);
            PlacementResult<ServerStatus, PlacementKey> result =
                    selectRole(request, metadata, role, group, routing);
            if (result.status() != PlacementResult.Status.SUCCESS) {
                Logger.debug(
                        "Failed to select {} worker for request {}",
                        role.getCode(),
                        context.data().getRequestId());
                return switch (result.status()) {
                    case REJECTED, BLOCKED, ENDPOINT_CHANGED ->
                            routing.validateEndpoints(result.asFailure());
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
        return routing.validateEndpoints(PlacementResult.success(routing.prepare()));
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
            RouteResult.Builder routing) {
        return switch (role) {
            case PREFILL, PDFUSION -> selectPrefill(request, metadata, role, group, routing);
            case DECODE -> selectDecode(request, metadata, group, routing);
            case ENCODER -> selectEncoder(request, group, routing);
            case VIT -> selectVit(request.getRequestId(), group, routing);
            case FRONTEND ->
                    throw new IllegalArgumentException(
                            "Endpoint selection is not supported for FRONTEND");
        };
    }

    private PlacementResult<ServerStatus, PlacementKey> selectEncoder(
            Request request, String group, RouteResult.Builder route) {
        var directory = endpoints.encoderRoutingSnapshot();
        var config = route.ctx().data().getConfig();
        var selection =
                encoderSelector.select(
                        directory,
                        group,
                        request.getEncoderCacheHitLen() != null,
                        config.isQueue()
                                ? config.getDispatcher().getMaxInflightPerEncoderWorker()
                                : Integer.MAX_VALUE);
        if (selection.status() != PlacementResult.Status.SUCCESS) return selection.asFailure();
        var selected = selection.value();
        com.google.common.base.Preconditions.checkArgument(
                directory.stream().anyMatch(candidate -> candidate == selected),
                "Encoder strategy must return an original candidate");
        WorkerEndpoint selectedEndpoint =
                endpoints.selectEndpoint(RoleType.ENCODER, selected.address());
        if (selectedEndpoint == null || selectedEndpoint != selected)
            return PlacementResult.endpointChanged(
                    new PlacementKey(RoleType.ENCODER, group, null), selection);
        var status = selectedEndpoint.getStatus();
        var assignment =
                ServerStatus.forWorker(
                        RoleType.ENCODER,
                        request.getRequestId(),
                        status.topologySnapshot(),
                        status.committedEngineObservation());
        route.add(selectedEndpoint, assignment, 0, 0);
        return PlacementResult.success(assignment);
    }

    private PlacementResult<ServerStatus, PlacementKey> selectVit(
            String requestId, String group, RouteResult.Builder route) {
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
        return PlacementResult.blocked(new PlacementKey(RoleType.VIT, group, null));
    }

    private PlacementResult<ServerStatus, PlacementKey> captureVit(
            String address, String group, String requestId, RouteResult.Builder route) {
        WorkerEndpoint selectedEndpoint = endpoints.selectEndpoint(RoleType.VIT, address);
        if (selectedEndpoint == null) {
            return PlacementResult.endpointChanged(
                    new PlacementKey(RoleType.VIT, group, null), PlacementResult.success(address));
        }
        var status = selectedEndpoint.getStatus();
        var topology = status.topologySnapshot();
        if (group != null && !group.equals(topology.group())) {
            return null;
        }
        var engine = status.committedEngineObservation();
        ServerStatus result = ServerStatus.forWorker(RoleType.VIT, requestId, topology, engine);
        return PlacementResult.success(route.add(selectedEndpoint, result, -1L, 0L));
    }

    private PlacementResult<ServerStatus, PlacementKey> selectPrefill(
            Request request,
            SchedulingMetadata metadata,
            RoleType role,
            String group,
            RouteResult.Builder route) {
        var directory = endpoints.prefillRoutingSnapshot(role);
        var result =
                prefillSelector.select(
                        request, metadata, role, group, directory, System::currentTimeMillis);
        if (result.diagnostics() != null) {
            var data = route.ctx().data();
            if (result.diagnostics().get("routingDecision")
                    instanceof org.flexlb.dao.pv.RoutingDecision decision) {
                data.recordRoutingDecision(decision);
            }
            if (result.diagnostics().get("cacheQuery")
                    instanceof org.flexlb.cache.domain.CacheMatchResult query) {
                data.recordCacheQuery(query.source().name(), query.queryTimeUs());
            }
        }
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
        WorkerEndpoint selectedEndpoint = endpoints.selectEndpoint(role, selected.address());
        if (selectedEndpoint == null
                || selectedEndpoint != endpoint
                || selectedEndpoint.getStatus().getGenerationId() != selected.generationId()) {
            return PlacementResult.endpointChanged(new PlacementKey(role, group, null), result);
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
        route.add(
                selectedEndpoint,
                assignment,
                selected.prefillWorkMs(),
                selected.ownershipVersion());
        route.ctx().data().recordCacheSelection(role, endpoint.getIp(), selected.cacheHitTokens());
        reportSelection(role, endpoint.getIp(), request.getSeqLen(), selected);
        if (selected.cacheMatches() != null) {
            try {
                cacheService.trackRoutingPrediction(
                        request.getRequestId(),
                        role,
                        group,
                        status,
                        request.getSeqLen(),
                        selected.cacheHitTokens(),
                        selected.cacheMatches());
            } catch (RuntimeException failure) {
                Logger.warn("Cache prediction observation failed", failure);
            }
        }
        return PlacementResult.success(assignment);
    }

    private void reportCacheAffinity(RoleType role, String ip, String reason) {
        if (reason == null) return;
        try {
            cacheReporter.reportCacheAffinityDecision(role, ip, reason);
        } catch (Throwable telemetryFailure) {
            Logger.warn("Cache affinity telemetry failed", telemetryFailure);
        }
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
            Request request, SchedulingMetadata metadata, String group, RouteResult.Builder route) {
        DecodeInputs cached = DECODE_INPUTS.get();
        DecodeInputs inputs = cached.inUse ? new DecodeInputs() : cached;
        inputs.inUse = true;
        try {
            endpoints.decodeRoutingSnapshot(group, inputs.decode);
            var result =
                    decodeSelector.select(
                            request, metadata, group, inputs.readOnly, capacityReader);
            if (result.status() != PlacementResult.Status.SUCCESS) {
                return result.asFailure();
            }
            var selected = result.value();
            WorkerEndpoint selectedEndpoint = endpoints.selectDecodeEndpoint(selected);
            if (selectedEndpoint == null) {
                return PlacementResult.endpointChanged(
                        new PlacementKey(RoleType.DECODE, group, null), result);
            }
            if (selectedEndpoint.getStatus().getGenerationId() != selected.generationId()) {
                return PlacementResult.endpointChanged(
                        new PlacementKey(RoleType.DECODE, group, null), result);
            }
            com.google.common.base.Preconditions.checkState(
                    selectedEndpoint instanceof DecodeEndpoint,
                    "Decode selection requires a Decode endpoint");
            ServerStatus assignment =
                    ServerStatus.forWorker(
                            RoleType.DECODE,
                            request.getRequestId(),
                            selected.topology(),
                            selected.workerStatus().fields());
            return PlacementResult.success(
                    route.add(selectedEndpoint, assignment, -1L, selected.admissionVersion()));
        } finally {
            int captured = inputs.decode.size();
            inputs.decode.clear();
            inputs.inUse = false;
            if (inputs == cached && captured > 4096) {
                DECODE_INPUTS.remove();
            }
        }
    }

    private DecodeResources.AdmissionSummary readDecodeCapacity(DecodeRoutingView view) {
        var selectedEndpoint = endpoints.selectDecodeEndpoint(view);
        return selectedEndpoint == null
                ? null
                : ((DecodeEndpoint) selectedEndpoint).admissionSummary();
    }

    private static final class DecodeInputs {
        final ArrayList<DecodeRoutingView> decode = new ArrayList<>();
        final List<DecodeRoutingView> readOnly = java.util.Collections.unmodifiableList(decode);
        boolean inUse;
    }
}
