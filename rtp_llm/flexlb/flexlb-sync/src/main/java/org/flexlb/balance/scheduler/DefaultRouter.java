package org.flexlb.balance.scheduler;

import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.policy.GroupRoutingPolicy;
import org.flexlb.balance.strategy.ConfiguredLoadBalanceSelector;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.balance.strategy.StaticCapacityExceededException;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

@Component
public class DefaultRouter implements Router {

    private final ConfiguredLoadBalanceSelector endpointSelector;
    private final GroupRoutingPolicy groupRoutingPolicy;
    private final List<RoleType> requiredRoles;

    public DefaultRouter(
            ConfiguredLoadBalanceSelector endpointSelector,
            GroupRoutingPolicy groupRoutingPolicy,
            ModelMetaConfig modelMetaConfig) {
        this.endpointSelector = java.util.Objects.requireNonNull(
                endpointSelector, "endpointSelector");
        this.groupRoutingPolicy = java.util.Objects.requireNonNull(
                groupRoutingPolicy, "groupRoutingPolicy");
        this.requiredRoles = List.copyOf(
                java.util.Objects.requireNonNull(
                        modelMetaConfig, "modelMetaConfig").requiredRoles());
    }

    @Override
    public Response routeDirect(BalanceContext context) {
        Response validationFailure = validateRequest(context);
        if (validationFailure != null) {
            return validationFailure;
        }
        try (PinnedRouting routing = selectAll(context, requiredRoles)) {
            if (!routing.success()) {
                return buildFailureResponse(routing.failedRole());
            }
            return commitDirect(context, routing.selections());
        } catch (StaticCapacityExceededException failure) {
            return buildStaticCapacityFailure(failure);
        }
    }

    @Override
    public QueueRoutingResult routeForQueue(BalanceContext context) {
        Response validationFailure = validateRequest(context);
        if (validationFailure != null) {
            return new QueueRoutingResult.Rejected(validationFailure);
        }
        try (PinnedRouting routing = selectAll(context, requiredRoles)) {
            if (!routing.success()) {
                return new QueueRoutingResult.Deferred(
                        routing.failedRole(), routing.failedGroup());
            }
            Response response = buildSuccessResponse(routing.serverStatuses());
            return QueueRouteAdmission.prepare(
                    context, routing.selections(), response);
        } catch (StaticCapacityExceededException failure) {
            return new QueueRoutingResult.Rejected(
                    buildStaticCapacityFailure(failure));
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
            List<RoleType> roles) {
        List<SelectedRole> selected = new ArrayList<>(roles.size());
        GroupRoutingDecision groupDecision = groupRoutingPolicy.route(context);
        String policyGroup = groupDecision.group();
        String group = policyGroup;
        if (groupDecision.hasGroup()) {
            Logger.debug(
                    "Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.getRequestId(),
                    groupDecision.policyName(),
                    group);
        }

        try {
            for (RoleType role : roles) {
                SelectedRole selection = endpointSelector.select(
                        context, role, group);
                if (selection == null) {
                    Logger.debug(
                            "No bindable {} worker in this routing attempt",
                            role.getCode());
                    return new PinnedRouting(selected, role, group);
                }
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

    private Response commitDirect(BalanceContext context, List<SelectedRole> selections) {
        List<DirectOwner> owners = new ArrayList<>(selections.size());
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
                        PrefillWorkLedger.DirectRegistration registration =
                                prefill.registerDirectRequest(
                                        pin, context.getRequestId(), selected.prefillWorkMs());
                        try {
                            owners.add(new DirectPrefillOwner(pin, registration));
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
                            owners.add(new DirectDecodeOwner(pin, decode, reservation));
                        } catch (RuntimeException | Error appendFailure) {
                            rollbackDecodeReservation(
                                    decode, reservation, appendFailure);
                            throw appendFailure;
                        }
                        pin = null;
                    } else {
                        owners.add(new StatelessDirectOwner(pin));
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
            for (DirectOwner owner : owners) {
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
            List<DirectOwner> owners,
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
            List<DirectOwner> owners,
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
            PrefillWorkLedger.DirectRegistration registration,
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
            endpoint.rollbackExact(reservation);
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

    private static Response buildStaticCapacityFailure(
            StaticCapacityExceededException failure) {
        Response response = Response.error(
                StrategyErrorType.RESOURCE_EXHAUSTED,
                AdmissionRejectReason.RESOURCE_EXHAUSTED);
        response.setErrorMessage(
                StrategyErrorType.RESOURCE_EXHAUSTED.buildErrorMessage(
                        failure.getMessage()));
        return response;
    }

    private static final class PinnedRouting implements AutoCloseable {
        private final List<SelectedRole> selections;
        private final RoleType failedRole;
        private final String failedGroup;

        private PinnedRouting(
                List<SelectedRole> selections,
                RoleType failedRole,
                String failedGroup) {
            this.selections = selections;
            this.failedRole = failedRole;
            this.failedGroup = failedGroup;
        }

        private boolean success() {
            return failedRole == null;
        }

        private RoleType failedRole() {
            return failedRole;
        }

        private String failedGroup() {
            return failedGroup;
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

    private sealed interface DirectOwner {
        void commit();

        void rollback();

        void closePin();
    }

    private record DirectPrefillOwner(
            WorkerEndpoint.GenerationPin pin,
            PrefillWorkLedger.DirectRegistration registration)
            implements DirectOwner {
        @Override
        public void commit() {
            registration.commit();
        }

        @Override
        public void rollback() {
            registration.close();
        }

        @Override
        public void closePin() {
            pin.close();
        }
    }

    private record DirectDecodeOwner(
            WorkerEndpoint.GenerationPin pin,
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation)
            implements DirectOwner {
        @Override
        public void commit() {
            // The canonical Decode registry already owns this exact handle.
        }

        @Override
        public void rollback() {
            endpoint.rollbackExact(reservation);
        }

        @Override
        public void closePin() {
            pin.close();
        }
    }

    private record StatelessDirectOwner(
            WorkerEndpoint.GenerationPin pin) implements DirectOwner {
        @Override
        public void commit() {
            // Stateless roles publish only frozen response metadata.
        }

        @Override
        public void rollback() {
            // No role-local ownership was created.
        }

        @Override
        public void closePin() {
            pin.close();
        }
    }
}
