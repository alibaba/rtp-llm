package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
import org.flexlb.service.VitCacheSelector;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
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
        String group = policyGroup;
        try (PinnedRouting routing = new PinnedRouting(new ArrayList<>(requiredRoles.size()))) {
            if (request.getSelectedVit() != null) {
                WorkerAssignment vit = vitCacheSelector == null || !requiredRoles.contains(RoleType.VIT)
                        ? null : vitCacheSelector.selectPinned(context, policyGroup);
                if (vit == null) {
                    return PlacementResult.rejected(Response.error(StrategyErrorType.VIT_ROUTE_STALE));
                }
                routing.selections().add(vit);
                group = vit.group();
            }
            for (RoleType role : requiredRoles) {
                if (role == RoleType.VIT && request.getSelectedVit() != null) { continue; }
                PlacementResult<WorkerAssignment, RoleType> result = selectRole(request, metadata, role, group);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    Logger.debug("Failed to select {} worker for request {}", role.getCode(), context.data().getRequestId());
                    return switch (result.status()) {
                        case REJECTED -> PlacementResult.rejected(result.failure(), result.diagnostics());
                        case BLOCKED -> PlacementResult.blocked(new PlacementKey(result.blocker(), group, null),
                                result.failure(), result.diagnostics());
                        default -> throw new IllegalStateException("unexpected selector result: " + result.status());
                    };
                }
                WorkerAssignment selection = result.value();
                try {
                    routing.selections().add(selection);
                } catch (RuntimeException | Error appendFailure) {
                    Failures.append(appendFailure, Failures.close(selection));
                    throw appendFailure;
                }
                if (StringUtils.isBlank(policyGroup)) {
                    group = selection.group();
                }
            }
            var result = PlacementResult.<RequestRoute, PlacementKey>success(RequestRoute.prepare(
                    context, routing.selections()));
            routing.selections().clear(); // The completed result now owns both stateful selections.
            return result;
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

    private PlacementResult<WorkerAssignment, RoleType> selectRole(
            Request request, SchedulingMetadata metadata, RoleType role, String group) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    prefillSelector.select(request, metadata, role, group);
            case DECODE -> decodeSelector.select(request, metadata, group);
            case VIT -> selectedOrBlocked(
                    vitSelector.select(request.getRequestId(), role, group), role);
            case FRONTEND -> throw new IllegalArgumentException(
                    "Endpoint selection is not supported for FRONTEND");
        };
    }

    private static PlacementResult<WorkerAssignment, RoleType> selectedOrBlocked(
            WorkerAssignment selected, RoleType role) {
        return selected == null
                ? PlacementResult.blocked(role)
                : PlacementResult.success(selected);
    }

    /** Owns selected generation pins until RequestRoute takes them or selection exits. */
    private record PinnedRouting(List<WorkerAssignment> selections) implements AutoCloseable {
        @Override
        public void close() {
            Throwable failure = null;
            for (int index = selections.size() - 1; index >= 0; index--) {
                failure = Failures.append(failure, Failures.close(selections.get(index)));
            }
            Failures.rethrow(failure, "route selection cleanup failed");
        }
    }
}
