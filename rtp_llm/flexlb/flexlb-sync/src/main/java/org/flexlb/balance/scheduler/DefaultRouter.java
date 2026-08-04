package org.flexlb.balance.scheduler;

import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.RoutingResult;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.service.monitor.RoutingQueueReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;

@Component
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy", "cacheAffinityFirstStrategy"})
public class DefaultRouter implements Router {

    private final Map<RoleType, LoadBalancer> loadBalancerMap;
    private final RoutingQueueReporter routingQueueReporter;

    /**
     * Creates a router with the configured load-balancing strategy for each role type.
     *
     * @param configService provides the load-balancing configuration
     */
    public DefaultRouter(ConfigService configService, RoutingQueueReporter routingQueueReporter) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.loadBalancerMap = new EnumMap<>(RoleType.class);
        this.routingQueueReporter = routingQueueReporter;

        for (RoleType roleType : RoleType.values()) {
            LoadBalanceStrategyEnum strategy = config.getStrategyForRoleType(roleType);
            loadBalancerMap.put(roleType, LoadBalanceStrategyFactory.getLoadBalancer(strategy));
        }
    }

    /**
     * Routes a request to appropriate worker nodes based on model requirements and role types.
     *
     * <p>Roles are selected sequentially so that each successful worker's group constrains the
     * next selection. Any failed, cancelled, or errored route rolls back workers selected earlier
     * in the route.
     *
     * @param balanceContext the context containing request information and model details
     * @return a publisher that emits selected server statuses or an error response
     */
    @Override
    public Mono<Response> route(BalanceContext balanceContext) {
        return Mono.defer(() -> routeOnce(balanceContext));
    }

    private Mono<Response> routeOnce(BalanceContext balanceContext) {
        // 1. Validate request
        Response validationResponse = validateRequest(balanceContext);
        if (validationResponse != null) {
            return Mono.just(validationResponse);
        }

        // 2. Get routing configuration
        String requestId = balanceContext.getRequestId();
        ModelWorkerStatus workerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        List<RoleType> roleTypeList = workerStatus.getRoleTypeList();
        if (CollectionUtils.isEmpty(roleTypeList) || workerStatus.getWorkerTotalCount() == 0) {
            return Mono.just(Response.error(NO_AVAILABLE_WORKER));
        }

        RouteSelectionState selectionState = new RouteSelectionState();
        return routeNextRole(balanceContext, roleTypeList, 0, null, selectionState)
                .map(routingResult -> routingResult.success()
                        ? buildSuccessResponse(
                        routingResult.serverStatusList(),
                        () -> rollBackSelectedWorkers(
                                balanceContext, selectionState, "response_rollback"))
                        : buildFailureResponse(
                        rollBackAndReturn(balanceContext, routingResult, selectionState)))
                .doOnError(error -> rollBackSelectedWorkers(
                        balanceContext, selectionState, "route_error"))
                .doOnCancel(() -> rollBackSelectedWorkers(
                        balanceContext, selectionState, "route_cancelled"));
    }

    @Override
    public void rollBack(BalanceContext balanceContext, Response response) {
        if (response == null || !response.isSuccess() || CollectionUtils.isEmpty(response.getServerStatus())) {
            return;
        }
        Runnable rollbackAction = response.getRollbackAction();
        if (rollbackAction != null) {
            rollbackAction.run();
            return;
        }
        rollBackWorkers(balanceContext, response.getServerStatus(), "response_rollback");
    }

    /**
     * Validates the incoming request and checks model availability.
     *
     * @param balanceContext the context to validate
     * @return error response if validation fails, null if validation succeeds
     */
    private Response validateRequest(BalanceContext balanceContext) {
        if (balanceContext.getRequest() == null) {
            Logger.error("masterRequest is null");
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }

        return null;
    }

    /**
     * Selects the current role and continues with the next role after a successful selection.
     *
     * @param balanceContext request context shared by every role selection
     * @param roleTypeList   ordered role types to route
     * @param roleIndex      index of the role currently being selected
     * @param group          group selected for the previous role, or {@code null} for the first role
     * @param selectionState workers selected so far and their rollback state
     * @return a publisher that emits the aggregate routing result
     */
    private Mono<RoutingResult> routeNextRole(BalanceContext balanceContext,
                                              List<RoleType> roleTypeList,
                                              int roleIndex,
                                              String group,
                                              RouteSelectionState selectionState) {
        if (selectionState.isRollbackStarted()) {
            return Mono.empty();
        }
        if (roleIndex >= roleTypeList.size()) {
            return Mono.just(RoutingResult.success(selectionState.snapshot()));
        }
        RoleType roleType = roleTypeList.get(roleIndex);
        LoadBalancer loadBalancer = getLoadBalancer(roleType);
        return loadBalancer.select(balanceContext, roleType, group)
                .flatMap(serverStatus -> {
                    if (!serverStatus.isSuccess()) {
                        Logger.warn("Failed to select {} worker: {}", roleType.getCode(), serverStatus.getMessage());
                        return Mono.just(RoutingResult.failure(
                                selectionState.snapshot(), roleType, serverStatus.getMessage()));
                    }
                    SelectionRecordResult recordResult = selectionState.record(serverStatus);
                    if (recordResult == SelectionRecordResult.NOT_RECORDED) {
                        rollBackWorkers(
                                balanceContext,
                                List.of(serverStatus),
                                "late_selection_after_rollback");
                        return Mono.empty();
                    }
                    if (recordResult == SelectionRecordResult.ALREADY_ROLLED_BACK) {
                        return Mono.empty();
                    }
                    return routeNextRole(
                            balanceContext,
                            roleTypeList,
                            roleIndex + 1,
                            serverStatus.getGroup(),
                            selectionState);
                })
                .switchIfEmpty(Mono.defer(() -> selectionState.isRollbackStarted()
                        ? Mono.empty()
                        : Mono.just(RoutingResult.failure(
                        selectionState.snapshot(),
                        roleType,
                        NO_AVAILABLE_WORKER.getErrorMsg()))));
    }

    /**
     * Get LoadBalancer based on role type
     */
    private LoadBalancer getLoadBalancer(RoleType roleType) {
        return loadBalancerMap.get(roleType);
    }

    private void rollBackSelectedWorkers(BalanceContext balanceContext,
                                         RouteSelectionState selectionState,
                                         String reason) {
        rollBackWorkers(balanceContext, selectionState.drainForRollback(), reason);
    }

    private void rollBackWorkers(BalanceContext balanceContext,
                                 List<ServerStatus> serverStatuses,
                                 String reason) {
        if (CollectionUtils.isEmpty(serverStatuses)) {
            return;
        }
        routingQueueReporter.reportRoutingRollback(reason, serverStatuses.size());
        Logger.info(String.format(
                "Routing rollback, requestId=%s, reason=%s, workerCount=%d, workers=%s",
                balanceContext.getRequestId(),
                reason,
                serverStatuses.size(),
                serverStatuses.stream()
                        .map(this::workerIdentifier)
                        .collect(Collectors.joining(","))));
        for (ServerStatus serverStatus : serverStatuses) {
            rollBackWorker(balanceContext, serverStatus);
        }
    }

    private String workerIdentifier(ServerStatus serverStatus) {
        return serverStatus.getRole() + "@" + serverStatus.getServerIp() + ":" + serverStatus.getHttpPort();
    }

    private void rollBackWorker(BalanceContext balanceContext, ServerStatus serverStatus) {
        String serverIpPort = serverStatus.getServerIp() + ":" + serverStatus.getHttpPort();
        String requestId = balanceContext.getRequestId();
        LoadBalancer loadBalancer = getLoadBalancer(serverStatus.getRole());
        loadBalancer.rollBack(serverIpPort, requestId);
    }

    private RoutingResult rollBackAndReturn(
            BalanceContext balanceContext,
            RoutingResult routingResult,
            RouteSelectionState selectionState) {
        rollBackSelectedWorkers(balanceContext, selectionState, "route_failure");
        return routingResult;
    }

    private enum SelectionRecordResult {
        RECORDED,
        NOT_RECORDED,
        ALREADY_ROLLED_BACK
    }

    private static class RouteSelectionState {

        private final Queue<ServerStatus> selectedWorkers = new ConcurrentLinkedQueue<>();
        private final AtomicBoolean rollbackStarted = new AtomicBoolean();

        private SelectionRecordResult record(ServerStatus serverStatus) {
            if (rollbackStarted.get()) {
                return SelectionRecordResult.NOT_RECORDED;
            }
            selectedWorkers.add(serverStatus);
            if (!rollbackStarted.get()) {
                return SelectionRecordResult.RECORDED;
            }
            return selectedWorkers.remove(serverStatus)
                    ? SelectionRecordResult.NOT_RECORDED
                    : SelectionRecordResult.ALREADY_ROLLED_BACK;
        }

        private List<ServerStatus> snapshot() {
            return selectedWorkers.stream()
                    .toList();
        }

        private List<ServerStatus> drainForRollback() {
            if (!rollbackStarted.compareAndSet(false, true)) {
                return List.of();
            }
            List<ServerStatus> workersToRollBack = new ArrayList<>();
            ServerStatus selectedWorker;
            while ((selectedWorker = selectedWorkers.poll()) != null) {
                workersToRollBack.add(selectedWorker);
            }
            return workersToRollBack;
        }

        private boolean isRollbackStarted() {
            return rollbackStarted.get();
        }
    }

    private Response buildSuccessResponse(List<ServerStatus> serverStatusList, Runnable rollbackAction) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(serverStatusList);
        response.setRollbackAction(rollbackAction);
        return response;
    }

    private Response buildFailureResponse(RoutingResult routingResult) {
        StrategyErrorType errorType = routingResult.failedRoleType().getErrorType();
        String detailMessage = routingResult.errorMessage();

        Response response = new Response();
        response.setSuccess(false);
        response.setCode(errorType.getErrorCode());
        response.setErrorMessage(errorType.getErrorMsg() + ": " + detailMessage);
        return response;
    }
}
