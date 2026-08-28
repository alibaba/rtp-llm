package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.strategy.CostBasedDecodeStrategy;
import org.flexlb.balance.strategy.CostBasedPrefillStrategy;
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
import org.springframework.stereotype.Component;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.ArrayList;
import java.util.List;

@Component
public class DefaultRouter {

    private final CostBasedPrefillStrategy prefillSelector;
    private final CostBasedDecodeStrategy decodeSelector;
    private final RandomStrategy vitSelector;
    private final ConfigService configService;
    private final List<RoleType> requiredRoles;
    private final PlacementAvailability placementAvailability;

    public DefaultRouter(
            CostBasedPrefillStrategy prefillSelector,
            CostBasedDecodeStrategy decodeSelector,
            RandomStrategy vitSelector,
            ConfigService configService,
            ModelMetaConfig modelMetaConfig) {
        this(prefillSelector, decodeSelector, vitSelector,
                configService,
                modelMetaConfig,
                new PlacementAvailability());
    }

    @Autowired
    public DefaultRouter(
            CostBasedPrefillStrategy prefillSelector,
            CostBasedDecodeStrategy decodeSelector,
            RandomStrategy vitSelector,
            ConfigService configService,
            ModelMetaConfig modelMetaConfig,
            PlacementAvailability placementAvailability) {
        this.prefillSelector = java.util.Objects.requireNonNull(
                prefillSelector, "prefillSelector");
        this.decodeSelector = java.util.Objects.requireNonNull(
                decodeSelector, "decodeSelector");
        this.vitSelector = java.util.Objects.requireNonNull(
                vitSelector, "vitSelector");
        this.configService = java.util.Objects.requireNonNull(
                configService, "configService");
        this.requiredRoles = List.copyOf(
                java.util.Objects.requireNonNull(
                        modelMetaConfig, "modelMetaConfig").requiredRoles());
        this.placementAvailability = java.util.Objects.requireNonNull(
                placementAvailability, "placementAvailability");
    }

    public Response routeDirect(BalanceContext context) {
        Response validationFailure = validateRequest(context);
        if (validationFailure != null) {
            return validationFailure;
        }
        try (PinnedRouting routing = selectAll(
                context, requiredRoles, false)) {
            if (routing.rejection() != null) {
                return routing.rejection();
            }
            if (!routing.success()) {
                return buildFailureResponse(routing.failure().role());
            }
            return commitDirect(context, routing.selections());
        }
    }

    public PlacementResult<QueueRouteAdmission, PlacementKey> routeForQueue(
            BalanceContext context) {
        Response validationFailure = validateRequest(context);
        if (validationFailure != null) {
            return PlacementResult.rejected(validationFailure);
        }
        try (PinnedRouting routing = selectAll(
                context, requiredRoles, true)) {
            if (routing.rejection() != null) {
                return PlacementResult.rejected(routing.rejection());
            }
            if (!routing.success()) {
                return PlacementResult.blocked(routing.failure());
            }
            Response response = buildSuccessResponse(routing.serverStatuses());
            return PlacementResult.success(
                    QueueRouteAdmission.prepare(
                            context,
                            routing.selections(),
                            response,
                            (exactContext, group) -> decodeSelector
                                    .select(
                                            exactContext, RoleType.DECODE, group),
                            placementAvailability));
        }
    }

    private Response validateRequest(BalanceContext context) {
        if (context == null || context.getRequest() == null) {
            Logger.error("masterRequest is null");
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        return null;
    }

    private PinnedRouting selectAll(
            BalanceContext context,
            List<RoleType> roles,
            boolean queueSelection) {
        List<SelectedRole> selected = new ArrayList<>(roles.size());
        String policyGroup = resolvePolicyGroup(context);
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
                PlacementResult<SelectedRole, RoleType> result = queueSelection
                        ? queueSelection(context, role, group)
                        : directSelection(context, role, group);
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

    private String resolvePolicyGroup(BalanceContext context) {
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

    private PlacementResult<SelectedRole, RoleType> directSelection(
            BalanceContext context, RoleType role, String group) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    selectedOrBlocked(
                            prefillSelector.select(context, role, group), role);
            case DECODE -> decodeSelector.select(context, role, group);
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

    private PlacementResult<SelectedRole, RoleType> queueSelection(
            BalanceContext context, RoleType role, String group) {
        return switch (role) {
            case PREFILL, PDFUSION ->
                    prefillSelector.selectForQueue(context, role, group);
            case DECODE ->
                    decodeSelector.select(context, role, group);
            case VIT -> directSelection(context, role, group);
            case FRONTEND -> throw new IllegalArgumentException(
                    "Endpoint selection is not supported for FRONTEND");
        };
    }

    private Response commitDirect(BalanceContext context, List<SelectedRole> selections) {
        List<DirectOwnership> owners = new ArrayList<>(selections.size());
        Response response;
        try {
            for (SelectedRole selected : selections) {
                if (selected.serverStatus().getRequestId() != context.getRequestId()) {
                    throw new IllegalStateException(
                            "selected role belongs to another DIRECT request");
                }
                WorkerEndpoint.GenerationPin pin = selected.takeGenerationPin();
                try {
                    WorkerEndpoint endpoint = pin.endpoint();
                    RoleType role = selected.serverStatus().getRole();
                    if (role == RoleType.PREFILL || role == RoleType.PDFUSION) {
                        if (!(endpoint instanceof PrefillEndpoint prefill)) {
                            throw new IllegalStateException(
                                    "Prefill selection has another endpoint type");
                        }
                        PrefillState.DirectRegistration registration =
                                prefill.registerDirectRequest(
                                        pin, context.getRequestId(), selected.prefillWorkMs());
                        try {
                            owners.add(new DirectOwnership(pin, registration));
                        } catch (RuntimeException | Error appendFailure) {
                            closeDirectRegistration(
                                    registration, appendFailure);
                            throw appendFailure;
                        }
                        pin = null;
                    } else if (role == RoleType.DECODE) {
                        if (!(endpoint instanceof DecodeEndpoint decode)) {
                            throw new IllegalStateException(
                                    "Decode selection has another endpoint type");
                        }
                        long sequenceLength = Math.max(0L, context.getRequest().getSeqLen());
                        long expectedKv =
                                context.getConfig()
                                        .decodeKvReservationTokens(
                                                sequenceLength,
                                                context.getRequest().getMaxNewTokens(),
                                                selected.decodeTotalKv());
                        DecodeEndpoint.ReservationHandle reservation =
                                decode.reservePinned(
                                        pin,
                                        context.getRequestId(),
                                        sequenceLength,
                                        expectedKv,
                                        context.getPriority());
                        try {
                            owners.add(new DirectOwnership(pin, decode, reservation));
                        } catch (RuntimeException | Error appendFailure) {
                            rollbackDecodeReservation(
                                    decode, reservation, appendFailure);
                            throw appendFailure;
                        }
                        pin = null;
                    } else {
                        owners.add(new DirectOwnership(pin));
                        pin = null;
                    }
                } catch (RuntimeException | Error leafFailure) {
                    if (pin != null) {
                        closeGenerationPin(pin, leafFailure);
                    }
                    throw leafFailure;
                }
            }
            response = buildSuccessResponse(serverStatuses(selections));
            // Every commit leaf below is a same-thread, allocation-free ownership
            // move. All fallible registration and response construction has
            // already completed, so this loop cannot partially commit legally.
            for (DirectOwnership owner : owners) {
                owner.commit();
            }
        } catch (RuntimeException | Error failure) {
            rollbackDirect(owners, failure);
            closeOwnerPins(owners, failure);
            throw failure;
        }
        Throwable closeFailure = closeOwnerPins(owners, null);
        if (closeFailure != null) {
            throw propagate(closeFailure);
        }
        return response;
    }

    private static Throwable rollbackDirect(
            List<DirectOwnership> owners,
            Throwable primaryFailure) {
        Throwable failure = primaryFailure;
        for (int index = owners.size() - 1; index >= 0; index--) {
            try {
                owners.get(index).rollback();
            } catch (Throwable rollbackFailure) {
                failure = appendFailure(failure, rollbackFailure);
            }
        }
        return failure;
    }

    private static Throwable closeOwnerPins(
            List<DirectOwnership> owners,
            Throwable primaryFailure) {
        Throwable failure = primaryFailure;
        for (int index = owners.size() - 1; index >= 0; index--) {
            try {
                owners.get(index).closePin();
            } catch (Throwable closeFailure) {
                failure = appendFailure(failure, closeFailure);
            }
        }
        return failure;
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

    private static Throwable closeDirectRegistration(
            PrefillState.DirectRegistration registration,
            Throwable primaryFailure) {
        try {
            registration.close();
        } catch (Throwable closeFailure) {
            return appendFailure(primaryFailure, closeFailure);
        }
        return primaryFailure;
    }

    private static Throwable rollbackDecodeReservation(
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation,
            Throwable primaryFailure) {
        try {
            endpoint.releaseReservationExact(reservation);
        } catch (Throwable rollbackFailure) {
            return appendFailure(primaryFailure, rollbackFailure);
        }
        return primaryFailure;
    }

    private static Throwable closeGenerationPin(
            WorkerEndpoint.GenerationPin pin,
            Throwable primaryFailure) {
        try {
            pin.close();
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
                "DIRECT route cleanup failed", failure);
    }

    private static Response buildSuccessResponse(
            List<ServerStatus> statuses) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(statuses);
        return response;
    }

    private static Response buildFailureResponse(RoleType failedRole) {
        StrategyErrorType errorType = failedRole == null
                ? NO_AVAILABLE_WORKER : failedRole.getErrorType();
        Response response = new Response();
        response.setSuccess(false);
        response.setCode(errorType.getErrorCode());
        response.setErrorMessage(errorType.getErrorMsg());
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

    private record DirectOwnership(
            WorkerEndpoint.GenerationPin pin,
            PrefillState.DirectRegistration prefill,
            DecodeEndpoint decode,
            DecodeEndpoint.ReservationHandle decodeReservation) {

        private DirectOwnership {
            java.util.Objects.requireNonNull(pin, "pin");
        }

        private DirectOwnership(WorkerEndpoint.GenerationPin pin) {
            this(pin, null, null, null);
        }

        private DirectOwnership(
                WorkerEndpoint.GenerationPin pin,
                PrefillState.DirectRegistration prefill) {
            this(pin, java.util.Objects.requireNonNull(prefill, "prefill"),
                    null, null);
        }

        private DirectOwnership(
                WorkerEndpoint.GenerationPin pin,
                DecodeEndpoint decode,
                DecodeEndpoint.ReservationHandle reservation) {
            this(pin, null,
                    java.util.Objects.requireNonNull(decode, "decode"),
                    java.util.Objects.requireNonNull(reservation, "reservation"));
        }

        private void commit() {
            if (prefill != null) {
                prefill.commit();
            }
        }

        private void rollback() {
            if (prefill != null) {
                prefill.close();
            } else if (decode != null) {
                decode.releaseReservationExact(decodeReservation);
            }
        }

        private void closePin() {
            pin.close();
        }
    }
}
