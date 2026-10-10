package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.VitWorkerSelector;
import org.flexlb.balance.strategy.WorkerAssignment;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;
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
        RequestRequirements requestInputs = context == null ? null : context.getRequirements();
        if (requestInputs == null) {
            Logger.error("request has no registered inputs");
            return PlacementResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        if (StringUtils.isNotBlank(policyGroup)) {
            Logger.info("Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.getRequestId(), "trafficPolicy", policyGroup);
        }
        String group = policyGroup;
        try (PinnedRouting routing = new PinnedRouting(new ArrayList<>(requiredRoles.size()))) {
            for (RoleType role : requiredRoles) {
                PlacementResult<WorkerAssignment, RoleType> result = selectRole(requestInputs, context.getConfig(), role, group);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    Logger.debug("Failed to select {} worker for request {}", role.getCode(), context.getRequestId());
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

    String resolvePolicyGroup(RequestContext context) {
        if (context == null || context.getRequirements() == null) {
            return null;
        }
        var selector = context.getConfig().getRouter().getGroupSelector();
        RequestRequirements request = context.getRequirements();
        return selector == null ? null : selector.resolveTargetGroup(
                request.requestId(), request.apiKey(), request.seqLen()).orElse(null);
    }

    private PlacementResult<WorkerAssignment, RoleType> selectRole(
            RequestRequirements requestInputs, FlexlbConfig config, RoleType role, String group) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    prefillSelector.select(requestInputs, config, role, group);
            case DECODE -> decodeSelector.select(requestInputs, group);
            case VIT -> selectedOrBlocked(
                    vitSelector.select(requestInputs.requestId(), role, group), role);
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
