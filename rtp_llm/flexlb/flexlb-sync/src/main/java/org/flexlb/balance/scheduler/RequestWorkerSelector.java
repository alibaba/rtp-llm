package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.flexlb.util.Failures;
import org.flexlb.service.VitCacheSelector;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Objects;

@Component
public class RequestWorkerSelector {

    private final CostBasedPrefillStrategy prefillSelector;
    private final DecodeSelector decodeSelector;
    private final VitWorkerSelector vitSelector;
    private final List<RoleType> requiredRoles;
    @Autowired
    private VitCacheSelector vitCacheSelector;

    @Autowired
    public RequestWorkerSelector(
            CostBasedPrefillStrategy prefillSelector,
            DecodeSelector decodeSelector,
            VitWorkerSelector vitSelector,
            ModelMetaConfig modelMetaConfig) {
        this.prefillSelector = Objects.requireNonNull(
                prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(
                decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(
                vitSelector, "vitSelector");
        this.requiredRoles = List.copyOf(
                Objects.requireNonNull(
                        modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    public PlacementResult<RequestRoute, PlacementKey> select(RequestContext context, String policyGroup) {
        Request request = context == null ? null : context.data().getRequest();
        SchedulingMetadata metadata = context == null ? null : context.data().getSchedulingMetadata();
        if (request == null || metadata == null) {
            Logger.error("request has no scheduling inputs");
            return PlacementResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        if (StringUtils.isNotBlank(policyGroup)) {
            Logger.info("Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.data().getRequestId(), "trafficPolicy", policyGroup);
        }
        RequestRoute routing = new RequestRoute(context);
        boolean complete = false;
        Throwable primary = null;
        try {
            String group = policyGroup;
            if (request.getSelectedVit() != null) {
                ServerStatus vit = vitCacheSelector == null || !requiredRoles.contains(RoleType.VIT)
                        ? null : vitCacheSelector.selectPinned(context, policyGroup, routing);
                if (vit == null) {
                    return PlacementResult.rejected(Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                group = vit.getGroup();
            }
            for (RoleType role : requiredRoles) {
                if (role == RoleType.VIT && request.getSelectedVit() != null) { continue; }
                PlacementResult<ServerStatus, RoleType> result = selectRole(request, metadata, role, group, routing);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    Logger.debug("Failed to select {} worker for request {}", role.getCode(), context.data().getRequestId());
                    return switch (result.status()) {
                        case REJECTED -> PlacementResult.rejected(result.failure(), result.diagnostics());
                        case BLOCKED -> PlacementResult.blocked(new PlacementKey(result.blocker(), group, null),
                                result.failure(), result.diagnostics());
                        default -> throw new IllegalStateException("unexpected selector result: " + result.status());
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
                if (primary != null) { Failures.append(primary, cleanup); }
                else { Failures.rethrow(cleanup, "route selection cleanup failed"); }
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
        if (vit.isSuccess()) { response.setServerStatus(List.of(vit)); }
        return response;
    }

    public boolean selectedVitIsValid(RequestContext context) {
        if (context.data().getRequest().getSelectedVit() == null) { return true; }
        return vitCacheSelector != null && requiredRoles.contains(RoleType.VIT)
                && vitCacheSelector.validate(context, resolvePolicyGroup(context)).isSuccess();
    }

    String resolvePolicyGroup(RequestContext context) {
        if (context == null || context.data().getRequest() == null) {
            return null;
        }
        var selector = context.data().getConfig().getRouter().getGroupSelector();
        Request request = context.data().getRequest();
        return selector == null ? null : selector.resolveTargetGroup(
                request.getRequestId(), request.getApiKey(), request.getSeqLen()).orElse(null);
    }

    private PlacementResult<ServerStatus, RoleType> selectRole(
            Request request, SchedulingMetadata metadata, RoleType role, String group, RequestRoute routing) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    prefillSelector.select(request, metadata, role, group, routing);
            case DECODE -> decodeSelector.select(request, metadata, group, routing);
            case VIT -> selectedOrBlocked(
                    vitSelector.select(request.getRequestId(), role, group, routing), role);
            case FRONTEND -> throw new IllegalArgumentException(
                    "Endpoint selection is not supported for FRONTEND");
        };
    }

    private static PlacementResult<ServerStatus, RoleType> selectedOrBlocked(
            ServerStatus selected, RoleType role) {
        return selected == null
                ? PlacementResult.blocked(role)
                : PlacementResult.success(selected);
    }

}
