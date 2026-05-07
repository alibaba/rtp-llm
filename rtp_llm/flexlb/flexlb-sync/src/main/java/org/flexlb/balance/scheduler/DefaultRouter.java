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
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;

@Component
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy"})
public class DefaultRouter implements Router {

    private static final int MAX_SELECT_COUNT = 1000;

    private final ConfigService configService;
    private final Map<RoleType, LoadBalancer> loadBalancerMap;

    public DefaultRouter(ConfigService configService) {
        this.configService = configService;
        FlexlbConfig config = configService.loadBalanceConfig();
        this.loadBalancerMap = new EnumMap<>(RoleType.class);

        for (RoleType roleType : RoleType.values()) {
            LoadBalanceStrategyEnum strategy = config.getStrategyForRoleType(roleType);
            loadBalancerMap.put(roleType, LoadBalanceStrategyFactory.getLoadBalancer(strategy));
        }
    }

    /**
     * Routes a request to appropriate worker nodes based on model requirements and role types.
     *
     * <p>This method implements the core routing logic for load balancing across different
     * worker types (Prefill, Decode, PDFusion, VIT).
     *
     * @param balanceContext the context containing request information and model details
     * @return Response containing selected server statuses or error information
     */
    @Override
    public Response route(BalanceContext balanceContext) {
        long startTimeInMicros = System.nanoTime() / 1000;
        // 1. Validate request
        Response validationResponse = validateRequest(balanceContext);
        if (validationResponse != null) {
            return validationResponse;
        }

        // 2. Get routing configuration
        long requestId = balanceContext.getRequestId();
        ModelWorkerStatus workerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        List<RoleType> roleTypeList = workerStatus.getRoleTypeList();
        if (CollectionUtils.isEmpty(roleTypeList)) {
            return Response.error(NO_AVAILABLE_WORKER);
        }

        // 3. Execute routing decision
        RoutingResult routingResult = routeByRoleType(balanceContext, roleTypeList);

        // 4. Build response based on routing result
        Response response;
        if (routingResult.success()) {
            response = buildSuccessResponse(requestId, routingResult.serverStatusList());
        } else {
            rollBackRoutingFailure(balanceContext, routingResult);
            response = buildFailureResponse(requestId, routingResult);
        }

        return response;
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

        if (EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS == null) {
            Logger.error("targetModelRoleWorkerStatus is null");
            return Response.error(NO_AVAILABLE_WORKER);
        }

        return null;
    }

    /**
     * Execute routing decision, select optimal server for each role type
     *
     * @param balanceContext Routing context
     * @param roleTypeList List of required role types
     * @return Routing result
     */
    public RoutingResult routeByRoleType(BalanceContext balanceContext, List<RoleType> roleTypeList) {
        List<ServerStatus> serverStatusList = new ArrayList<>();
        String group = null;

        for (RoleType roleType : roleTypeList) {
            LoadBalancer loadBalancer = getLoadBalancer(roleType);
            ServerStatus serverStatus = loadBalancer.select(balanceContext, roleType, group);

            if (!serverStatus.isSuccess()) {
                // Selection failed, return failure result
                Logger.warn("Failed to select {} worker: {}", roleType.getCode(), serverStatus.getMessage());
                return RoutingResult.failure(serverStatusList, roleType, serverStatus.getMessage());
            }

            // Record server selection metrics
            serverStatusList.add(serverStatus);

            // Update group for affinity-based selection of subsequent roles
            group = serverStatus.getGroup();
        }

        return RoutingResult.success(serverStatusList);
    }

    /**
     * Get LoadBalancer based on role type
     */
    private LoadBalancer getLoadBalancer(RoleType roleType) {
        return loadBalancerMap.get(roleType);
    }

    /**
     * Rollback handling for routing failure
     * If partial roles succeeded but subsequent roles failed, rollback local incremental updates for previously selected roles
     *
     * @param balanceContext Routing context
     * @param routingResult Routing result
     */
    private void rollBackRoutingFailure(BalanceContext balanceContext, RoutingResult routingResult) {

        List<ServerStatus> partialResults = routingResult.serverStatusList();
        for (ServerStatus serverStatus : partialResults) {
            String serverIpPort = serverStatus.getServerIp() + ":" + serverStatus.getHttpPort();
            long requestId = balanceContext.getRequestId();

            RoleType role = serverStatus.getRole();
            LoadBalancer loadBalancer = getLoadBalancer(role);
            loadBalancer.rollBack(serverIpPort, requestId);
        }
    }

    private Response buildSuccessResponse(long requestId, List<ServerStatus> serverStatusList) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(serverStatusList);
        return response;
    }

    private Response buildFailureResponse(long requestId, RoutingResult routingResult) {
        StrategyErrorType errorType = routingResult.failedRoleType().getErrorType();
        String detailMessage = routingResult.errorMessage();

        Response response = new Response();
        response.setSuccess(false);
        response.setCode(errorType.getErrorCode());
        response.setErrorMessage(errorType.getErrorMsg() + ": " + detailMessage);
        return response;
    }

    /**
     * Select up to N healthy workers of the given role, sorted by available concurrency.
     *
     * <p>Read-only query; does not register in-flight or update load balancer state.
     * Returned list is shuffled so multiple callers don't all hit list[0] for the same worker.
     *
     * @param balanceContext routing context (used for audit/logging only)
     * @param role           target role
     * @param count          desired count; -1 means "all healthy" (capped at MAX_SELECT_COUNT)
     * @return Response carrying selected ServerStatus list, ttlMs, version, totalWorkers
     */
    public Response selectNWorkers(BalanceContext balanceContext, RoleType role, int count) {
        if (count == 0 || count < -1) {
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        if (role == null) {
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }

        ModelWorkerStatus mws = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        Map<String, WorkerStatus> roleMap = mws.getRoleStatusMap(role);
        if (roleMap == null || roleMap.isEmpty()) {
            return Response.error(role.getErrorType());
        }

        long totalHealthy = roleMap.values().stream()
                .filter(WorkerStatus::isAlive)
                .count();
        if (totalHealthy == 0) {
            return Response.error(role.getErrorType());
        }

        int effectiveCount = (count == -1)
                ? Math.min((int) totalHealthy, MAX_SELECT_COUNT)
                : Math.min(count, MAX_SELECT_COUNT);

        List<ServerStatus> selected = roleMap.values().stream()
                .filter(WorkerStatus::isAlive)
                .sorted(Comparator.comparingLong((WorkerStatus ws) ->
                        ws.getAvailableConcurrency() == null ? 0L : ws.getAvailableConcurrency()
                ).reversed())
                .limit(effectiveCount)
                .map(ws -> toServerStatus(ws, role))
                .collect(Collectors.toList());

        Collections.shuffle(selected);

        Response resp = new Response();
        resp.setSuccess(true);
        resp.setServerStatus(selected);
        resp.setTtlMs(configService.loadBalanceConfig().getSelectWorkersTtlMs());
        resp.setVersion(EngineWorkerStatus.getCurrentVersion());
        resp.setTotalWorkers((int) totalHealthy);
        return resp;
    }

    private ServerStatus toServerStatus(WorkerStatus ws, RoleType role) {
        ServerStatus s = new ServerStatus();
        s.setRole(role);
        s.setServerIp(ws.getIp());
        s.setHttpPort(ws.getPort());
        s.setGrpcPort(CommonUtils.toGrpcPort(ws.getPort()));
        s.setSuccess(true);
        return s;
    }
}
