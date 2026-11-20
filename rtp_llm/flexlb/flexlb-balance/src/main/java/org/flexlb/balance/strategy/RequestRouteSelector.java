package org.flexlb.balance.strategy;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.request.Request;
import org.flexlb.balance.request.Response;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestPhase;
import org.flexlb.balance.schedule.RequestRoute;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerAdmissionCapacity;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.status.ServerStatus;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Selects and registers the required roles, releasing partial registrations when the combination fails.
 * Complete combinations transfer their resources to the scheduler for validation, Context association and commit.
 */
@Component
public class RequestRouteSelector {
    private final PrefillSelectionStrategy prefillStrategy;
    private final DecodeStrategy decodeStrategy;
    private final VitStrategy vitStrategy;
    private final EncoderStrategy encoderStrategy;
    private final List<RoleType> requiredRoles;
    private final EndpointRegistry endpoints;

    @Autowired
    public RequestRouteSelector(
            EndpointRegistry endpoints,
            PrefillSelectionStrategy prefillStrategy,
            DecodeStrategy decodeStrategy,
            VitStrategy vitStrategy,
            EncoderStrategy encoderStrategy,
            ModelMetaConfig modelMetaConfig) {
        this.endpoints = Objects.requireNonNull(endpoints, "endpoints");
        this.encoderStrategy = Objects.requireNonNull(encoderStrategy);
        this.prefillStrategy = Objects.requireNonNull(prefillStrategy, "prefillStrategy");
        this.decodeStrategy = Objects.requireNonNull(decodeStrategy, "decodeStrategy");
        this.vitStrategy = Objects.requireNonNull(vitStrategy, "vitStrategy");
        this.requiredRoles =
                List.copyOf(
                        Objects.requireNonNull(modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    public List<WorkerAdmissionCapacity> requestCapacities() {
        var result = new ArrayList<WorkerAdmissionCapacity>();
        for (var role : requiredRoles)
            for (var endpoint : endpoints.<WorkerEndpoint>endpoints(role, null))
                result.add(requestCapacity(endpoint));
        return result;
    }

    public WorkerAdmissionCapacity requestCapacity(WorkerEndpoint endpoint) {
        if (endpoint instanceof org.flexlb.worker.endpoint.PrefillEndpoint prefill)
            return prefill.requestCapacity();
        if (endpoint instanceof org.flexlb.worker.endpoint.DecodeEndpoint decode)
            return decodeStrategy.requestCapacity(decode);
        if (endpoint instanceof org.flexlb.worker.endpoint.EncoderEndpoint encoder)
            return encoderStrategy.requestCapacity(encoder);
        var status = endpoint.getStatus();
        // VIT has no local request-count reservation; permit one availability probe.
        return new WorkerAdmissionCapacity(
                CapacityKey.exact(status.getRole(), status.getGroup(), endpoint.ipPort()),
                status.getGenerationId(),
                status.committedWorkerStatus().cursor().statusVersion(),
                endpoint.acceptsNewRequests() ? 1L : 0L);
    }

    public SelectionResult<RequestRoute, CapacityKey> selectAndRegister(
            RequestContext context, String policyGroup) {
        return selectAndRegister(context, policyGroup, null);
    }

    public SelectionResult<RequestRoute, CapacityKey> selectAndRegister(
            RequestContext context, String policyGroup, RoleSelection reclaimed) {
        Request request = context == null ? null : context.getRequest();
        SchedulingMetadata metadata = context == null ? null : context.getSchedulingMetadata();
        if (request == null || metadata == null) {
            if (reclaimed != null) rollbackSelection(reclaimed);
            return SelectionResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        if (StringUtils.isNotBlank(policyGroup)) {
            Logger.info(
                    "Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.getRequestId(),
                    "trafficPolicy",
                    policyGroup);
        }
        var selections = new ArrayList<RoleSelection>(4);
        boolean transferred = false;
        Throwable selectionFailure = null;
        try {
            String group = policyGroup;
            if (reclaimed != null) {
                selections.add(reclaimed);
                if (!reclaimed.endpoint().acceptsNewRequests()
                        || StringUtils.isNotBlank(policyGroup)
                                && !policyGroup.equals(reclaimed.group()))
                    return SelectionResult.retry();
                group = reclaimed.group();
            }
            if (request.getSelectedVit() != null) {
                RoleSelection vit =
                        !requiredRoles.contains(RoleType.VIT)
                                ? null
                                : vitStrategy.selectForRoute(context, group);
                if (vit == null) {
                    return SelectionResult.rejected(
                            Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                selections.add(vit);
                group = vit.group();
            }
            var requested = context.getRequestedRoles();
            if ((!requested.isEmpty() && !requiredRoles.containsAll(requested))
                    || context.getPhase() == RequestPhase.ENCODER
                            && !requiredRoles.contains(RoleType.ENCODER)) {
                return SelectionResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
            }
            var roles =
                    requested.isEmpty()
                            ? (context.getPhase() == RequestPhase.ENCODER
                                    ? List.of(RoleType.ENCODER)
                                    : requiredRoles)
                            : requiredRoles.stream().filter(requested::contains).toList();
            for (RoleType role : roles) {
                if (context.getFuture() != null && context.getFuture().isDone()
                        || context.snapshot().cancellationReason() != null)
                    return SelectionResult.closed();
                if (reclaimed != null && role == reclaimed.role()) continue;
                if (role == RoleType.VIT && request.getSelectedVit() != null) {
                    continue;
                }
                context.beginRoutingAttempt(role);
                var result =
                        switch (role) {
                            case PREFILL -> prefillStrategy.selectAndRegister(context, group);
                            case PDFUSION ->
                                    prefillStrategy.selectAndRegisterFusion(context, group);
                            case DECODE -> decodeStrategy.selectAndRegister(context, group);
                            case ENCODER -> encoderStrategy.selectAndRegister(context, group);
                            case VIT -> vitStrategy.selectAndRegister(context, group);
                            default ->
                                    throw new IllegalArgumentException(
                                            "Endpoint selection is not supported for " + role);
                        };
                if (result.status() != SelectionResult.Status.SUCCESS) {
                    Logger.debug(
                            "Failed to select {} worker for request {}",
                            role.getCode(),
                            context.getRequestId());
                    var unavailable = RequestRoute.firstUnavailableEndpoint(selections);
                    return unavailable == null
                            ? result.asFailure()
                            : SelectionResult.retry(
                                    Response.error(
                                            StrategyErrorType.forRole(
                                                    unavailable.getStatus().getRole())));
                }
                RoleSelection selection = result.value();
                try {
                    selections.add(selection);
                } catch (RuntimeException | Error failure) {
                    Failures.append(
                            failure, Failures.run(null, () -> rollbackSelection(selection)));
                    throw failure;
                }
                if (StringUtils.isBlank(policyGroup)) {
                    group = selection.group();
                }
            }
            if (context.getFuture() != null && context.getFuture().isDone()
                    || context.snapshot().cancellationReason() != null)
                return SelectionResult.closed();
            var execution = new RequestRoute(context, selections);
            var result = SelectionResult.<RequestRoute, CapacityKey>success(execution);
            transferred = true;
            return result;
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

    public Response routeVit(RequestContext context) {
        if (context == null || context.getRequest() == null) {
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        if (!requiredRoles.contains(RoleType.VIT)) {
            return Response.error(StrategyErrorType.NO_VIT_WORKER);
        }
        ServerStatus vit = vitStrategy.selectCache(context, resolvePolicyGroup(context));
        Response response = new Response();
        response.setSuccess(vit.isSuccess());
        response.setCode(vit.getCode());
        response.setErrorMessage(vit.getMessage());
        if (vit.isSuccess()) {
            response.setServerStatus(List.of(vit));
        }
        return response;
    }

    public String resolvePolicyGroup(RequestContext context) {
        if (context == null || context.getRequest() == null) {
            return null;
        }
        var selector = context.getConfig().getRouter().getGroupSelector();
        Request request = context.getRequest();
        return selector == null
                ? null
                : selector.resolveTargetGroup(
                                request.getRequestId(), request.getApiKey(), request.getSeqLen())
                        .orElse(null);
    }

    public boolean selectedVitIsValid(RequestContext context) {
        if (context.getRequest().getSelectedVit() == null) {
            return true;
        }
        return requiredRoles.contains(RoleType.VIT)
                && vitStrategy.validateSelection(context, resolvePolicyGroup(context)) != null;
    }

    public java.util.OptionalLong estimateQueuedTokens(RequestContext request, String group) {
        if (request.getPhase() != org.flexlb.balance.schedule.RequestPhase.GENERATION)
            return java.util.OptionalLong.empty();
        var role = requiredRoles.contains(RoleType.PREFILL) ? RoleType.PREFILL : RoleType.PDFUSION;
        return prefillStrategy.estimateQueuedTokens(request.getRequest(), role, group);
    }

    public WorkerEndpoint blockedEndpoint(CapacityKey key) {
        return key == null || key.endpoint() == null
                ? null
                : endpoints.getEndpoint(key.role(), key.endpoint());
    }

    private static void rollback(List<RoleSelection> selections) {
        Throwable failure = null;
        for (var role : selections) failure = Failures.run(failure, () -> rollbackSelection(role));
        Failures.rethrow(failure, "selection registration rollback failed");
    }

    private static void rollbackSelection(RoleSelection selection) {
        if (selection.registration() != null) {
            ((org.flexlb.worker.endpoint.SelectionRollback) selection.endpoint())
                    .rollbackSelection(selection.registration());
        }
    }
}
