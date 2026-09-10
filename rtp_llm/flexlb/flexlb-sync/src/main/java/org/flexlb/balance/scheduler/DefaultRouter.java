package org.flexlb.balance.scheduler;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.ScheduledRequest.DecodeBinding;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
import org.flexlb.balance.strategy.DecodeSelector;
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
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

@Component
public class DefaultRouter {

    private final CostBasedPrefillStrategy prefillSelector;
    private final DecodeSelector decodeSelector;
    private final RandomStrategy vitSelector;
    private final ConfigService configService;
    private final List<RoleType> requiredRoles;
    private final RoleType queueAdmissionRole;

    @Autowired
    public DefaultRouter(
            CostBasedPrefillStrategy prefillSelector,
            DecodeSelector decodeSelector,
            RandomStrategy vitSelector,
            ConfigService configService,
            ModelMetaConfig modelMetaConfig) {
        this.prefillSelector = Objects.requireNonNull(
                prefillSelector, "prefillSelector");
        this.decodeSelector = Objects.requireNonNull(
                decodeSelector, "decodeSelector");
        this.vitSelector = Objects.requireNonNull(
                vitSelector, "vitSelector");
        this.configService = Objects.requireNonNull(
                configService, "configService");
        this.requiredRoles = List.copyOf(
                Objects.requireNonNull(
                        modelMetaConfig, "modelMetaConfig").requiredRoles());
        this.queueAdmissionRole = requiredRoles.stream()
                .filter(role -> role == RoleType.PREFILL
                        || role == RoleType.PDFUSION)
                .findFirst()
                .orElse(RoleType.PREFILL);
    }

    public PlacementResult<RouteAdmission, PlacementKey> select(BalanceContext context) {
        return select(context, resolvePolicyGroup(context));
    }

    public PlacementResult<RouteAdmission, PlacementKey> select(BalanceContext context, String policyGroup) {
        Response validationFailure = validateRequest(context);
        if (validationFailure != null) { return PlacementResult.rejected(validationFailure); }
        DecodeBinding decodeAdmission = DecodeBinding.capture(context);
        try (PinnedRouting routing = selectAll(context, requiredRoles, policyGroup, decodeAdmission)) {
            if (routing.rejection() != null) { return PlacementResult.rejected(routing.rejection()); }
            if (!routing.success()) { return PlacementResult.blocked(routing.failure()); }
            return PlacementResult.success(RouteAdmission.prepare(context, routing.selections(),
                    buildSuccessResponse(routing.serverStatuses()), decodeAdmission));
        }
    }

    /** Capacity domain that gates publication into the selected Prefill queue. */
    RoleType queueAdmissionRole() {
        return queueAdmissionRole;
    }

    private Response validateRequest(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            Logger.error("masterRequest is null");
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        return null;
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
            for (RoleType role : roles) {
                PlacementResult<SelectedRole, RoleType> result =
                        selectRole(context, role, group, decodeAdmission);
                if (result.status() != PlacementResult.Status.SUCCESS) {
                    Logger.debug(
                            "Failed to select {} worker for request {}",
                            role.getCode(), context.getRequestId());
                    if (result.status() == PlacementResult.Status.REJECTED) {
                        return new PinnedRouting(
                                selected, null, result.rejection());
                    }
                    if (result.status() == PlacementResult.Status.BLOCKED) {
                        return new PinnedRouting(
                                selected,
                                new PlacementKey(result.blocker(), group),
                                null);
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
            return new PinnedRouting(selected, null, null);
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
            case DECODE -> decodeSelector.select(decodeAdmission, group);
            case VIT -> selectedOrBlocked(
                    vitSelector.select(context, role, group), role);
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
        private final PlacementKey failure;
        private final Response rejection;

        private PinnedRouting(
                List<SelectedRole> selections,
                PlacementKey failure,
                Response rejection) {
            this.selections = selections;
            this.failure = failure;
            this.rejection = rejection;
        }

        private boolean success() {
            return failure == null && rejection == null;
        }

        private PlacementKey failure() {
            return failure;
        }

        private Response rejection() {
            return rejection;
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
