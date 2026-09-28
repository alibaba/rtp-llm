package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
import org.flexlb.balance.strategy.EncoderStrategy;
import org.flexlb.balance.strategy.RandomStrategy;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.flexlb.service.VitCacheSelector;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

@Component
public class DefaultRouter {

    private final CostBasedPrefillStrategy prefillSelector;
    private final DecodeSelector decodeSelector;
    private final RandomStrategy vitSelector;
    private final EncoderStrategy encoderSelector;
    private final ConfigService configService;
    private final List<RoleType> requiredRoles;
    @Autowired
    private VitCacheSelector vitCacheSelector;

    @Autowired
    public DefaultRouter(
            CostBasedPrefillStrategy prefillSelector,
            DecodeSelector decodeSelector,
            RandomStrategy vitSelector,
            EncoderStrategy encoderSelector,
            ConfigService configService,
            ModelMetaConfig modelMetaConfig) {
        this.prefillSelector = Objects.requireNonNull(prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(vitSelector, "vitSelector");
        this.encoderSelector = Objects.requireNonNull(encoderSelector, "encoderSelector");
        this.configService = Objects.requireNonNull(configService, "configService");
        this.requiredRoles = List.copyOf(Objects.requireNonNull(modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    boolean isEncoderOnly(BalanceContext context) {
        return requestedRoles(context).equals(List.of(RoleType.ENCODER));
    }

    PlacementResult<SelectedRole, PlacementKey> selectEncoder(BalanceContext context) {
        Response failure = validateRequest(context);
        if (failure != null) {
            return PlacementResult.rejected(failure);
        }
        if (!isEncoderOnly(context)) {
            return PlacementResult.rejected(Response.error(StrategyErrorType.INVALID_REQUEST));
        }
        SelectedRole selected = encoderSelector.select(context, resolvePolicyGroup(context));
        return selected == null
                ? PlacementResult.blocked(new PlacementKey(RoleType.ENCODER, resolvePolicyGroup(context)))
                : PlacementResult.success(selected);
    }

    public PlacementResult<RouteAdmission, PlacementKey> select(BalanceContext context) {
        return select(context, resolvePolicyGroup(context));
    }

    public PlacementResult<RouteAdmission, PlacementKey> select(BalanceContext context, String policyGroup) {
        Response validationFailure = validateRequest(context);
        if (validationFailure != null) { return PlacementResult.rejected(validationFailure); }
        DecodeBinding decodeAdmission = DecodeBinding.capture(context);
        try (PinnedRouting routing = selectAll(context, requestedRoles(context), policyGroup, decodeAdmission)) {
            if (routing.blocker() != null) {
                return PlacementResult.blocked(routing.blocker(), routing.failure(), routing.diagnostics);
            }
            if (routing.failure() != null) {
                return PlacementResult.rejected(routing.failure(), routing.diagnostics);
            }
            return PlacementResult.success(RouteAdmission.prepare(context, routing.selections(),
                    buildSuccessResponse(routing.serverStatuses()), decodeAdmission));
        }
    }

    public Response routeVit(BalanceContext context) {
        Response invalid = validateRequest(context);
        if (invalid != null) { return invalid; }
        if (!requiredRoles.contains(RoleType.VIT) || vitCacheSelector == null) {
            return Response.error(StrategyErrorType.NO_VIT_WORKER);
        }
        ServerStatus vit = vitCacheSelector.select(context, resolvePolicyGroup(context));
        if (vit.isSuccess()) { return buildSuccessResponse(List.of(vit)); }
        Response failure = new Response();
        failure.setSuccess(false);
        failure.setCode(vit.getCode());
        failure.setErrorMessage(vit.getMessage());
        return failure;
    }

    public boolean selectedVitIsValid(BalanceContext context) {
        if (context.getRequest() == null || context.getRequest().getSelectedVit() == null) { return true; }
        return vitCacheSelector != null && requiredRoles.contains(RoleType.VIT)
                && vitCacheSelector.validate(context, resolvePolicyGroup(context)).isSuccess();
    }

    private Response validateRequest(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            Logger.error("masterRequest is null");
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        Set<RoleType> requested = context.getRequestedRoles();
        if (requested != null && !requiredRoles.containsAll(requested)) {
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        return null;
    }

    private List<RoleType> requestedRoles(BalanceContext context) {
        Set<RoleType> requested = context.getRequestedRoles();
        return requested == null ? requiredRoles
                : requiredRoles.stream().filter(requested::contains).toList();
    }

    private PinnedRouting selectAll(BalanceContext context, List<RoleType> roles, String policyGroup,
                                   DecodeBinding decodeAdmission) {
        List<SelectedRole> selected = new ArrayList<>(roles.size());
        String group = policyGroup;
        if (StringUtils.isNotBlank(policyGroup)) {
            Logger.info(
                    "Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.getRequestId(),
                    "trafficPolicy",
                    group);
        }

        try {
            if (context.getRequest().getSelectedVit() != null) {
                if (vitCacheSelector == null) {
                    return new PinnedRouting(selected, null, Response.error(StrategyErrorType.VIT_ROUTE_STALE), null);
                }
                SelectedRole vit = vitCacheSelector.selectPinned(context, policyGroup);
                if (vit == null) {
                    return new PinnedRouting(selected, null, Response.error(StrategyErrorType.VIT_ROUTE_STALE), null);
                }
                selected.add(vit);
                group = vit.serverStatus().getGroup();
            }
            for (RoleType role : roles) {
                if (role == RoleType.VIT && context.getRequest().getSelectedVit() != null) { continue; }
                PlacementResult<SelectedRole, RoleType> result =
                        selectRole(context, role, group, decodeAdmission);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    Logger.debug(
                            "Failed to select {} worker for request {}",
                            role.getCode(), context.getRequestId());
                    if (result.status() == PlacementResult.Status.REJECTED) {
                        return new PinnedRouting(
                                selected, null, result.failure(), result.diagnostics());
                    }
                    if (result.status() == PlacementResult.Status.BLOCKED) {
                        return new PinnedRouting(
                                selected,
                                new PlacementKey(result.blocker(), group),
                                result.failure(), result.diagnostics());
                    }
                    throw new IllegalStateException("unexpected selector result: "
                            + result.status());
                }
                SelectedRole selection = result.value();
                try {
                    selected.add(selection);
                } catch (RuntimeException | Error appendFailure) {
                    closeSelection(selection, appendFailure);
                    throw appendFailure;
                }
                if (StringUtils.isBlank(policyGroup)) {
                    group = selection.serverStatus().getGroup();
                }
            }
            return new PinnedRouting(selected, null, null, null);
        } catch (RuntimeException | Error failure) {
            closeSelections(selected, failure);
            throw failure;
        }
    }

    String resolvePolicyGroup(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            return null;
        }
        FlexlbConfig config = context.getConfig() != null
                ? context.getConfig()
                : configService.loadBalanceConfig();
        if (config == null || config.getRouter().getGroupSelector() == null) {
            return null;
        }
        return config.getRouter().getGroupSelector()
                .resolveTargetGroup(context.getRequest())
                .orElse(null);
    }

    private PlacementResult<SelectedRole, RoleType> selectRole(
            BalanceContext context, RoleType role, String group, DecodeBinding decodeAdmission) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    prefillSelector.select(context, role, group);
            case DECODE -> decodeSelector.select(context, decodeAdmission, group);
            case VIT -> selectedOrBlocked(
                    vitSelector.select(context, role, group), role);
            case ENCODER -> selectedOrBlocked(
                    encoderSelector.select(context, group), role);
            case FRONTEND -> throw new IllegalArgumentException(
                    "Endpoint selection is not supported for FRONTEND");
        };
    }

    private static PlacementResult<SelectedRole, RoleType> selectedOrBlocked(
            SelectedRole selected, RoleType role) {
        return selected == null
                ? PlacementResult.blocked(role)
                : PlacementResult.success(selected);
    }

    private static List<ServerStatus> serverStatuses(List<SelectedRole> selections) {
        List<ServerStatus> statuses = new ArrayList<>(selections.size());
        for (SelectedRole selection : selections) {
            statuses.add(selection.serverStatus());
        }
        return statuses;
    }

    private static Throwable closeSelections(
            List<SelectedRole> selections,
            Throwable primaryFailure) {
        Throwable failure = primaryFailure;
        for (int index = selections.size() - 1; index >= 0; index--) {
            failure = closeSelection(selections.get(index), failure);
        }
        return failure;
    }

    private static Throwable closeSelection(
            SelectedRole selection,
            Throwable primaryFailure) {
        try {
            selection.close();
        } catch (Throwable closeFailure) {
            return appendFailure(primaryFailure, closeFailure);
        }
        return primaryFailure;
    }

    private static Throwable appendFailure(
            Throwable primaryFailure,
            Throwable cleanupFailure) {
        if (primaryFailure == null) {
            return cleanupFailure;
        }
        if (primaryFailure != cleanupFailure) {
            primaryFailure.addSuppressed(cleanupFailure);
        }
        return primaryFailure;
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException(
                "route selection cleanup failed", failure);
    }

    private static Response buildSuccessResponse(
            List<ServerStatus> statuses) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(statuses);
        return response;
    }

    private static final class PinnedRouting implements AutoCloseable {
        private final List<SelectedRole> selections;
        private final PlacementKey blocker;
        private final Response failure;
        private final Map<String, Object> diagnostics;

        private PinnedRouting(
                List<SelectedRole> selections,
                PlacementKey blocker,
                Response failure, Map<String, Object> diagnostics) {
            this.selections = selections;
            this.blocker = blocker;
            this.failure = failure;
            this.diagnostics = diagnostics;
        }

        private PlacementKey blocker() {
            return blocker;
        }

        private Response failure() {
            return failure;
        }

        private List<SelectedRole> selections() {
            return selections;
        }

        private List<ServerStatus> serverStatuses() {
            return DefaultRouter.serverStatuses(selections);
        }

        @Override
        public void close() {
            Throwable failure = closeSelections(selections, null);
            if (failure != null) {
                throw propagate(failure);
            }
        }
    }

}
