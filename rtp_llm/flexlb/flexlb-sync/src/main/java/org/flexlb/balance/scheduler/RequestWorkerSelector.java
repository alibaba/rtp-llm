package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.EncoderSelection;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.VitCacheSelector;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;
import java.util.Set;

@Component
public class RequestWorkerSelector {

    private final CostBasedPrefillStrategy prefillSelector;
    private final DecodeSelector decodeSelector;
    private final VitWorkerSelector vitSelector;
    private final List<RoleType> requiredRoles;

    @Autowired(required = false)
    private EncoderStrategy encoderSelector;

    @Autowired private VitCacheSelector vitCacheSelector;

    @Autowired
    public RequestWorkerSelector(
            CostBasedPrefillStrategy prefillSelector,
            DecodeSelector decodeSelector,
            VitWorkerSelector vitSelector,
            ModelMetaConfig modelMetaConfig) {
        this.prefillSelector = Objects.requireNonNull(prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(vitSelector, "vitSelector");
        this.requiredRoles =
                List.copyOf(
                        Objects.requireNonNull(modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    boolean hasEncoderRole() {
        return requiredRoles.contains(RoleType.ENCODER);
    }

    boolean isEncoderOnly(RequestContext context) {
        if (context == null) {
            return false;
        }
        Set<RoleType> requested = context.data().getRequestedRoles();
        return requiredRoles.stream()
                .filter(role -> requested == null || requested.contains(role))
                .collect(java.util.stream.Collectors.toSet())
                .equals(Set.of(RoleType.ENCODER));
    }

    Response validateRoles(RequestContext context) {
        if (context == null || context.data().getRequest() == null) {
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        Set<RoleType> requested = context.data().getRequestedRoles();
        return requested != null && !requiredRoles.containsAll(requested)
                ? Response.error(StrategyErrorType.INVALID_REQUEST)
                : null;
    }

    PlacementResult<EncoderSelection, PlacementKey> selectEncoder(RequestContext context) {
        Response invalid = validateRoles(context);
        if (invalid != null || !isEncoderOnly(context)) {
            return PlacementResult.rejected(
                    invalid == null ? Response.error(StrategyErrorType.INVALID_REQUEST) : invalid);
        }
        String group = resolvePolicyGroup(context);
        EncoderSelection selected =
                encoderSelector == null ? null : encoderSelector.select(context.data(), group);
        return selected == null
                ? PlacementResult.blocked(
                        new PlacementKey(RoleType.ENCODER, group, null),
                        Response.error(StrategyErrorType.NO_ENCODER_WORKER),
                        null)
                : PlacementResult.success(selected);
    }

    public PlacementResult<RequestRoute, PlacementKey> select(
            RequestContext context, String policyGroup) {
        Response invalid = validateRoles(context);
        if (invalid != null) {
            return PlacementResult.rejected(invalid);
        }
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
                                : vitCacheSelector.selectPinned(
                                        context.data(), policyGroup, routing);
                if (vit == null) {
                    return PlacementResult.rejected(
                            Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                group = vit.getGroup();
            }
            for (RoleType role : requiredRoles) {
                var requested = context.data().getRequestedRoles();
                if (requested != null && !requested.contains(role)) {
                    continue;
                }
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
                        case REJECTED ->
                                PlacementResult.rejected(result.failure(), result.diagnostics());
                        case BLOCKED ->
                                PlacementResult.blocked(
                                        result.blocker(), result.failure(), result.diagnostics());
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
        ServerStatus vit = vitCacheSelector.select(context.data(), resolvePolicyGroup(context));
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
                && vitCacheSelector
                        .validate(context.data(), resolvePolicyGroup(context))
                        .isSuccess();
    }

    String resolvePolicyGroup(RequestContext context) {
        if (context == null || context.data().getRequest() == null) {
            return null;
        }
        var selector = context.data().getConfig().getRouter().getGroupSelector();
        Request request = context.data().getRequest();
        return selector == null ? null : selector.resolveTargetGroup(request).orElse(null);
    }

    private PlacementResult<ServerStatus, PlacementKey> selectRole(
            Request request,
            SchedulingMetadata metadata,
            RoleType role,
            String group,
            RequestRoute routing) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    prefillSelector.select(request, metadata, role, group, routing);
            case DECODE ->
                    withGroup(decodeSelector.select(routing.ctx().data(), group, routing), group);
            case VIT ->
                    withGroup(
                            selectedOrBlocked(
                                    vitSelector.select(
                                            request.getRequestId(), role, group, routing),
                                    role),
                            group);
            case ENCODER ->
                    withGroup(selectedEncoderForRoute(routing.ctx(), group, routing), group);
            case FRONTEND ->
                    throw new IllegalArgumentException(
                            "Endpoint selection is not supported for FRONTEND");
        };
    }

    private PlacementResult<ServerStatus, RoleType> selectedEncoderForRoute(
            RequestContext context, String group, RequestRoute routing) {
        try (EncoderSelection selected =
                encoderSelector == null ? null : encoderSelector.select(context.data(), group)) {
            return selectedOrBlocked(
                    selected == null ? null : selected.transferTo(routing), RoleType.ENCODER);
        }
    }

    private static PlacementResult<ServerStatus, PlacementKey> withGroup(
            PlacementResult<ServerStatus, RoleType> result, String group) {
        return new PlacementResult<>(
                result.status(),
                result.value(),
                result.failure(),
                result.blocker() == null ? null : new PlacementKey(result.blocker(), group, null),
                result.diagnostics());
    }

    private static PlacementResult<ServerStatus, RoleType> selectedOrBlocked(
            ServerStatus selected, RoleType role) {
        return selected == null ? PlacementResult.blocked(role) : PlacementResult.success(selected);
    }
}
