package org.flexlb.balance.strategy;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.schedule.RequestContext;
import org.flexlb.balance.schedule.RequestPhase;
import org.flexlb.balance.schedule.RequestResult;
import org.flexlb.balance.schedule.SchedulingMetadata;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.worker.endpoint.CapacityKey;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
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
    private final CostBasedPrefillStrategy prefillStrategy;
    private final DecodeStrategy decodeStrategy;
    private final VitStrategy vitStrategy;
    private final EncoderStrategy encoderStrategy;
    private final List<RoleType> requiredRoles;
    private final EndpointRegistry endpoints;

    @Autowired
    public RequestRouteSelector(
            EndpointRegistry endpoints,
            CostBasedPrefillStrategy prefillStrategy,
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

    public RequestResult select(RequestContext context, String policyGroup) {
        return select(context, policyGroup, null, null);
    }

    public RequestResult select(
            RequestContext context,
            String policyGroup,
            DecodeEndpoint reclaimedEndpoint,
            DecodeEndpoint.RequestRegistration reclaimedReservation) {
        Request request = context == null ? null : context.getRequest();
        SchedulingMetadata metadata = context == null ? null : context.getSchedulingMetadata();
        if (request == null || metadata == null) {
            if (reclaimedReservation != null)
                RoleSelection.releaseDecode(
                        reclaimedEndpoint,
                        reclaimedReservation,
                        WorkerEndpoint.ReleaseReason.LOCAL_ROLLBACK);
            Logger.error("request has no scheduling inputs");
            return RequestResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
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
            if (reclaimedReservation != null) {
                var reclaimed =
                        addDecode(
                                selections,
                                reclaimedEndpoint,
                                reclaimedReservation,
                                () -> {
                                    var view = reclaimedEndpoint.selectionSnapshot();
                                    return ServerStatus.forWorker(
                                            RoleType.DECODE,
                                            request.getRequestId(),
                                            view.topology(),
                                            view.workerStatus().fields());
                                });
                if (!reclaimedEndpoint.registrationState(reclaimedReservation).registered()
                        || StringUtils.isNotBlank(policyGroup)
                                && !policyGroup.equals(reclaimed.group()))
                    return RequestResult.retry();
                group = reclaimed.group();
            }
            if (request.getSelectedVit() != null) {
                RoleSelection vit =
                        !requiredRoles.contains(RoleType.VIT)
                                ? null
                                : vitStrategy.selectForRoute(context, group);
                if (vit == null) {
                    return RequestResult.rejected(
                            Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                selections.add(vit);
                group = vit.group();
            }
            var requested = context.getRequestedRoles();
            if ((!requested.isEmpty() && !requiredRoles.containsAll(requested))
                    || context.getPhase() == RequestPhase.ENCODER
                            && !requiredRoles.contains(RoleType.ENCODER)) {
                return RequestResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
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
                    return RequestResult.closed();
                if (role == RoleType.DECODE && reclaimedReservation != null) continue;
                if (role == RoleType.VIT && request.getSelectedVit() != null) {
                    continue;
                }
                context.beginRoutingAttempt(role);
                RoutingResult<RoleSelection, CapacityKey> result =
                        switch (role) {
                            case PREFILL, PDFUSION -> prefillStrategy.select(context, role, group);
                            case DECODE -> decodeStrategy.select(context, group);
                            case ENCODER -> encoderStrategy.select(context, group);
                            case VIT -> vitStrategy.select(context, group);
                            case FRONTEND ->
                                    throw new IllegalArgumentException(
                                            "Endpoint selection is not supported for FRONTEND");
                        };
                if (result.status() != RoutingResult.Status.SUCCESS) {
                    Logger.debug(
                            "Failed to select {} worker for request {}",
                            role.getCode(),
                            context.getRequestId());
                    return RequestResult.validateEndpoints(
                            selections, RequestResult.fromDecision(result));
                }
                RoleSelection selection = result.value();
                try {
                    selections.add(selection);
                } catch (RuntimeException | Error failure) {
                    Failures.append(failure, Failures.run(null, selection::rollbackSelection));
                    throw failure;
                }
                if (StringUtils.isBlank(policyGroup)) {
                    group = selection.group();
                }
            }
            if (context.getFuture() != null && context.getFuture().isDone()
                    || context.snapshot().cancellationReason() != null)
                return RequestResult.closed();
            var execution = new RequestResult(context, selections);
            transferred = true;
            return execution;
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
                && vitStrategy.validate(context, resolvePolicyGroup(context)).isSuccess();
    }

    public WorkerEndpoint blockedEndpoint(CapacityKey key) {
        return key == null || key.endpoint() == null
                ? null
                : endpoints.getEndpoint(key.role(), key.endpoint());
    }

    private static RoleSelection addDecode(
            List<RoleSelection> selections,
            DecodeEndpoint endpoint,
            DecodeEndpoint.RequestRegistration registration,
            java.util.function.Supplier<ServerStatus> metadata) {
        try {
            var selected = RoleSelection.decode(endpoint, metadata.get(), registration);
            selections.add(selected);
            return selected;
        } catch (RuntimeException | Error failure) {
            Failures.append(
                    failure,
                    Failures.run(
                            null,
                            () ->
                                    RoleSelection.releaseDecode(
                                            endpoint,
                                            registration,
                                            WorkerEndpoint.ReleaseReason.LOCAL_ROLLBACK)));
            throw failure;
        }
    }

    private static void rollback(List<RoleSelection> selections) {
        Throwable failure = null;
        for (var role : selections) failure = Failures.run(failure, role::rollbackSelection);
        Failures.rethrow(failure, "selection registration rollback failed");
    }
}
