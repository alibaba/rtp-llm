package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.RoutingResult;
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
import org.flexlb.dao.route.RequestPhase;
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

/**
 * Selects and registers the required roles, releasing partial registrations when the combination fails.
 */
@Component
public class RequestRouteSelector {
    private static final ThreadLocal<DecodeInputs> DECODE_INPUTS =
            ThreadLocal.withInitial(DecodeInputs::new);

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

    private final boolean direct;

    private final DecodeResources.AdmissionCapacity admissionCapacity;

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
        var configuration = configService.loadBalanceConfig();
        this.direct = configuration.isDirect();
        var availability = configuration.getRouter().getRoles().getDecode().getAvailability();
        this.admissionCapacity =
                new DecodeResources.AdmissionCapacity(
                        availability.getMaxEngineRequests() == null
                                ? 0
                                : availability.getMaxEngineRequests(),
                        availability.getMaxKvUsagePercent());
        this.deliveryMode = configService.loadBalanceConfig().getDispatcher().typeName();
        this.prefillSelector = Objects.requireNonNull(prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(vitSelector, "vitSelector");
        this.requiredRoles =
                List.copyOf(
                        Objects.requireNonNull(modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    public RoutingResult<RouteResult, CapacityKey> select(
            RequestContext context, String policyGroup) {
        return select(context, policyGroup, null, null);
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

    WorkerEndpoint blockedEndpoint(CapacityKey key) {
        return key == null || key.endpoint() == null
                ? null
                : endpoints.selectEndpoint(key.role(), key.endpoint());
    }

    RoutingResult<RouteResult, CapacityKey> select(
            RequestContext context,
            String policyGroup,
            DecodeEndpoint reclaimedEndpoint,
            DecodeResources.RequestReservation reclaimedReservation) {
        Request request = context == null ? null : context.data().getRequest();
        SchedulingMetadata metadata =
                context == null ? null : context.data().getSchedulingMetadata();
        if (request == null || metadata == null) {
            if (reclaimedReservation != null)
                RoleSelection.Decode.release(
                        reclaimedEndpoint,
                        reclaimedReservation,
                        DecodeResources.ReleaseReason.LOCAL_ROLLBACK);
            Logger.error("request has no scheduling inputs");
            return RoutingResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        if (StringUtils.isNotBlank(policyGroup)) {
            Logger.info(
                    "Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.data().getRequestId(),
                    "trafficPolicy",
                    policyGroup);
        }
        var selections = new ArrayList<RoleSelection>(4);
        boolean transferred = false;
        Throwable selectionFailure = null;
        try {
            String group = policyGroup;
            if (reclaimedReservation != null) {
                var reclaimed =
                        addDecode(
                                selections,
                                reclaimedEndpoint,
                                reclaimedReservation,
                                () -> {
                                    var view = reclaimedEndpoint.routingView();
                                    return ServerStatus.forWorker(
                                            RoleType.DECODE,
                                            request.getRequestId(),
                                            view.topology(),
                                            view.workerStatus().fields());
                                });
                if (!reclaimedEndpoint.ownership(reclaimedReservation).owned()
                        || StringUtils.isNotBlank(policyGroup)
                                && !policyGroup.equals(reclaimed.group()))
                    return RoutingResult.retry();
                group = reclaimed.group();
            }
            if (request.getSelectedVit() != null) {
                RoleSelection vit =
                        vitCacheSelector == null || !requiredRoles.contains(RoleType.VIT)
                                ? null
                                : vitCacheSelector.selectForRoute(context, group);
                if (vit == null) {
                    return RoutingResult.rejected(
                            Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                selections.add(vit);
                group = vit.group();
            }
            var requested = context.data().getRequestedRoles();
            if ((!requested.isEmpty() && !requiredRoles.containsAll(requested))
                    || context.data().getPhase() == RequestPhase.ENCODER
                            && !requiredRoles.contains(RoleType.ENCODER)) {
                return RoutingResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
            }
            var roles =
                    requested.isEmpty()
                            ? (context.data().getPhase() == RequestPhase.ENCODER
                                    ? List.of(RoleType.ENCODER)
                                    : requiredRoles)
                            : requiredRoles.stream().filter(requested::contains).toList();
            for (RoleType role : roles) {
                if (context.data().getFuture() != null && context.data().getFuture().isDone()
                        || context.snapshot().cancellation() != null) return RoutingResult.closed();
                if (role == RoleType.DECODE && reclaimedReservation != null) continue;
                if (role == RoleType.VIT && request.getSelectedVit() != null) {
                    continue;
                }
                context.data().beginRoutingAttempt(role);
                int previousSelections = selections.size();
                RoutingResult<?, CapacityKey> result =
                        switch (role) {
                            case PREFILL, PDFUSION ->
                                    selectPrefill(context, role, group, selections);
                            case DECODE -> selectDecode(context, group, selections);
                            case ENCODER -> selectEncoder(context, group, selections);
                            case VIT -> selectVit(request.getRequestId(), group, selections);
                            case FRONTEND ->
                                    throw new IllegalArgumentException(
                                            "Endpoint selection is not supported for FRONTEND");
                        };
                if (result.status() != RoutingResult.Status.SUCCESS) {
                    Logger.debug(
                            "Failed to select {} worker for request {}",
                            role.getCode(),
                            context.data().getRequestId());
                    return RouteResult.validateEndpoints(selections, result.asFailure());
                }
                com.google.common.base.Preconditions.checkState(
                        selections.size() == previousSelections + 1
                                && selections.getLast().role() == role,
                        "successful strategy must register exactly one %s selection",
                        role);
                RoleSelection selection = selections.getLast();
                if (StringUtils.isBlank(policyGroup)) {
                    group = selection.group();
                }
            }
            if (context.data().getFuture() != null && context.data().getFuture().isDone()
                    || context.snapshot().cancellation() != null) return RoutingResult.closed();
            var execution = new RouteResult(context, selections);
            var selected = execution.validateEndpoints(RoutingResult.success(execution));
            transferred = selected.status() == RoutingResult.Status.SUCCESS;
            return selected;
        } catch (RuntimeException | Error failure) {
            selectionFailure = failure;
            throw failure;
        } finally {
            if (!transferred) {
                var rollbackFailure = Failures.run(null, () -> rollback(selections));
                if (selectionFailure != null) Failures.append(selectionFailure, rollbackFailure);
                else Failures.rethrow(rollbackFailure, "selection registration rollback failed");
            }
        }
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

    private RoutingResult<?, CapacityKey> selectEncoder(
            RequestContext context, String group, List<RoleSelection> selections) {
        var directory = endpoints.encoderRoutingSnapshot();
        var request = context.data().getRequest();
        var selection =
                encoderSelector.select(
                        directory,
                        group,
                        request.getEncoderCacheHitLen() != null,
                        context.data().getRequestScheduler().encoderRequestLimit());
        if (selection.status() != RoutingResult.Status.SUCCESS) return selection.asFailure();
        var selected = selection.value();
        com.google.common.base.Preconditions.checkArgument(
                directory.stream().anyMatch(candidate -> candidate == selected),
                "Encoder strategy must return an original candidate");
        WorkerEndpoint selectedEndpoint =
                endpoints.selectEndpoint(RoleType.ENCODER, selected.address());
        if (selectedEndpoint == null || selectedEndpoint != selected)
            return RoutingResult.retry(Response.error(RoleType.ENCODER.getErrorType()));
        var status = selectedEndpoint.getStatus();
        var assignment =
                ServerStatus.forWorker(
                        RoleType.ENCODER,
                        request.getRequestId(),
                        status.topologySnapshot(),
                        status.committedEngineObservation());
        var role = new RoleSelection.Auxiliary(selectedEndpoint, assignment);
        selections.add(role);
        return selection;
    }

    private RoutingResult<RoleSelection, CapacityKey> selectVit(
            String requestId, String group, List<RoleSelection> selections) {
        List<String> choices = endpoints.endpointAddressSnapshot(RoleType.VIT);
        ArrayList<String> remaining = null;
        while (!choices.isEmpty()) {
            int index = vitSelector.selectIndex(choices);
            com.google.common.base.Preconditions.checkArgument(
                    index >= 0 && index < choices.size(), "VIT strategy returned an invalid index");
            var selected = captureVit(choices.get(index), group, requestId, selections);
            if (selected != null) return selected;
            if (remaining == null) {
                remaining = new ArrayList<>(choices);
                choices = java.util.Collections.unmodifiableList(remaining);
            }
            java.util.Collections.swap(remaining, index, remaining.size() - 1);
            remaining.removeLast();
        }
        return RoutingResult.blocked(new CapacityKey(RoleType.VIT, group, null));
    }

    private RoutingResult<RoleSelection, CapacityKey> captureVit(
            String address, String group, String requestId, List<RoleSelection> selections) {
        WorkerEndpoint selectedEndpoint = endpoints.selectEndpoint(RoleType.VIT, address);
        if (selectedEndpoint == null) {
            return RoutingResult.retry(Response.error(RoleType.VIT.getErrorType()));
        }
        var status = selectedEndpoint.getStatus();
        var topology = status.topologySnapshot();
        if (group != null && !group.equals(topology.group())) {
            return null;
        }
        var engine = status.committedEngineObservation();
        ServerStatus result = ServerStatus.forWorker(RoleType.VIT, requestId, topology, engine);
        var role = new RoleSelection.Auxiliary(selectedEndpoint, result);
        selections.add(role);
        return RoutingResult.success(role);
    }

    private RoutingResult<?, CapacityKey> selectPrefill(
            RequestContext context, RoleType role, String group, List<RoleSelection> selections) {
        var request = context.data().getRequest();
        var metadata = context.data().getSchedulingMetadata();
        var directory = endpoints.prefillRoutingSnapshot(role);
        var result =
                prefillSelector.selectAndRegister(
                        request,
                        metadata,
                        role,
                        group,
                        directory,
                        System::currentTimeMillis,
                        selected -> registerPrefill(context, role, selections, selected));
        if (result.diagnostics() != null) {
            var data = context.data();
            if (result.diagnostics().get("routingDecision")
                    instanceof org.flexlb.dao.pv.RoutingDecision decision) {
                data.recordRoutingDecision(decision);
            }
            if (result.diagnostics().get("cacheQuery")
                    instanceof org.flexlb.cache.domain.CacheMatchResult query) {
                data.recordCacheQuery(query.source().name(), query.queryTimeUs());
            }
        }
        if (result.status() != RoutingResult.Status.SUCCESS) {
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
        var status = endpoint.getStatus();
        context.data().recordCacheSelection(role, endpoint.getIp(), selected.cacheHitTokens());
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
        return result;
    }

    private <I extends PrefillSelectionStrategy.CandidateInput>
            RoutingResult.Registration registerPrefill(
                    RequestContext context,
                    RoleType role,
                    List<RoleSelection> selections,
                    PrefillSelectionStrategy.Selection<I> selected) {
        var request = context.data().getRequest();
        var candidate = endpoints.selectEndpoint(role, selected.address());
        if (!(candidate instanceof PrefillEndpoint endpoint)
                || endpoint.getStatus().getGenerationId() != selected.generationId())
            return RoutingResult.Registration.UNAVAILABLE;
        var assignment =
                ServerStatus.forWorker(
                        role,
                        request.getRequestId(),
                        endpoint.getStatus().topologySnapshot(),
                        endpoint.getStatus().committedEngineObservation());
        if (selected.projectedTtftMs() >= 0) assignment.setPrefillTime(selected.projectedTtftMs());
        var debug = new DebugInfo();
        debug.setHitCacheLen(selected.cacheHitTokens());
        assignment.setDebugInfo(debug);
        var data = context.data();
        data.initializeExecutionOrder(System.currentTimeMillis());
        var input =
                new org.flexlb.balance.planner.GroupPlanner.Item(
                        data.getRequestId(),
                        data.getPriority(),
                        data.getWorkerEnqueueSequence(),
                        data.getFirstWorkerEnqueueTime(),
                        data.getRequestExpiresAtMs(),
                        request.getHardKvTokens(),
                        selected.cacheHitTokens());
        var registered =
                endpoint.tryRegisterSelection(
                        input, selected.projectionInputs(), selected.prefillWorkMs(), !direct);
        if (registered.status() == RoutingResult.Registration.REGISTERED) {
            try {
                selections.add(
                        new RoleSelection.Prefill(
                                endpoint,
                                assignment,
                                selected.prefillWorkMs(),
                                registered.record()));
            } catch (RuntimeException | Error failure) {
                Failures.append(
                        failure, Failures.run(null, () -> endpoint.rollback(registered.record())));
                throw failure;
            }
        }
        return registered.status();
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

    private RoutingResult<?, CapacityKey> selectDecode(
            RequestContext context, String group, List<RoleSelection> selections) {
        var request = context.data().getRequest();
        var metadata = context.data().getSchedulingMetadata();
        DecodeInputs cached = DECODE_INPUTS.get();
        DecodeInputs inputs = cached.inUse ? new DecodeInputs() : cached;
        inputs.inUse = true;
        try {
            endpoints.decodeRoutingSnapshot(group, inputs.decode);
            return decodeSelector.selectAndRegister(
                    request,
                    metadata,
                    group,
                    inputs.readOnly,
                    capacityReader,
                    this::refreshDecode,
                    selected -> registerDecode(request, metadata, selections, selected));
        } finally {
            int captured = inputs.decode.size();
            inputs.decode.clear();
            inputs.inUse = false;
            if (inputs == cached && captured > 4096) {
                DECODE_INPUTS.remove();
            }
        }
    }

    private RoutingResult.Registration registerDecode(
            Request request,
            SchedulingMetadata metadata,
            List<RoleSelection> selections,
            DecodeRoutingView selected) {
        var candidate = endpoints.selectDecodeEndpoint(selected);
        if (!(candidate instanceof DecodeEndpoint endpoint))
            return RoutingResult.Registration.UNAVAILABLE;
        var acquired =
                endpoint.tryRegisterSelection(
                        selected,
                        request.getRequestId(),
                        request.getHardKvTokens(),
                        request.getExpectedKvTokens(),
                        metadata.priority(),
                        admissionCapacity);
        return switch (acquired.status()) {
            case VERSION_CONFLICT -> RoutingResult.Registration.CONFLICT;
            case ENDPOINT_RETIRED -> RoutingResult.Registration.UNAVAILABLE;
            case ALREADY_OWNED -> RoutingResult.Registration.CONFLICT;
            case CAPACITY_FULL -> RoutingResult.Registration.BLOCKED;
            case ACQUIRED -> {
                addDecode(
                        selections,
                        endpoint,
                        acquired.value(),
                        () ->
                                ServerStatus.forWorker(
                                        RoleType.DECODE,
                                        request.getRequestId(),
                                        selected.topology(),
                                        selected.workerStatus().fields()));
                yield RoutingResult.Registration.REGISTERED;
            }
        };
    }

    private static RoleSelection.Decode addDecode(
            List<RoleSelection> selections,
            DecodeEndpoint endpoint,
            DecodeResources.RequestReservation registration,
            java.util.function.Supplier<ServerStatus> metadata) {
        try {
            var selected = new RoleSelection.Decode(endpoint, metadata.get(), registration);
            selections.add(selected);
            return selected;
        } catch (RuntimeException | Error failure) {
            Failures.append(
                    failure,
                    Failures.run(
                            null,
                            () ->
                                    RoleSelection.Decode.release(
                                            endpoint,
                                            registration,
                                            DecodeResources.ReleaseReason.LOCAL_ROLLBACK)));
            throw failure;
        }
    }

    private DecodeRoutingView refreshDecode(DecodeRoutingView view) {
        var endpoint = endpoints.selectDecodeEndpoint(view);
        return endpoint instanceof DecodeEndpoint decode ? decode.routingView() : null;
    }

    private DecodeResources.AdmissionSummary readDecodeCapacity(DecodeRoutingView view) {
        var selectedEndpoint = endpoints.selectDecodeEndpoint(view);
        return selectedEndpoint == null
                ? null
                : ((DecodeEndpoint) selectedEndpoint).admissionSummary();
    }

    private static void rollback(List<RoleSelection> selections) {
        Throwable failure = null;
        for (var role : selections) {
            if (role instanceof RoleSelection.Prefill prefill && prefill.registration != null)
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        ((PrefillEndpoint) prefill.endpoint())
                                                .rollback(prefill.registration));
            else if (role instanceof RoleSelection.Decode decode)
                failure =
                        Failures.run(
                                failure,
                                () ->
                                        RoleSelection.Decode.release(
                                                (DecodeEndpoint) decode.endpoint(),
                                                decode.registration,
                                                DecodeResources.ReleaseReason.LOCAL_ROLLBACK));
        }
        Failures.rethrow(failure, "selection registration rollback failed");
    }

    private static final class DecodeInputs {
        final ArrayList<DecodeRoutingView> decode = new ArrayList<>();
        final List<DecodeRoutingView> readOnly = java.util.Collections.unmodifiableList(decode);
        boolean inUse;
    }
}
