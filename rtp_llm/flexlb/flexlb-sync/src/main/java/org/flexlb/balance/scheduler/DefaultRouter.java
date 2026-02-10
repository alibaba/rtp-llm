package org.flexlb.balance.scheduler;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.WhaleMasterConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.RoutingResult;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.Logger;
import org.springframework.context.annotation.DependsOn;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import static org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER;

@Component
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy"})
public class DefaultRouter implements Router {

    private final Map<RoleType, LoadBalancer> loadBalancerMap;

    public DefaultRouter(ConfigService configService) {
        WhaleMasterConfig config = configService.loadBalanceConfig();
        LoadBalanceStrategyEnum loadBalanceStrategyByConfig = config.getLoadBalanceStrategy();
        LoadBalanceStrategyEnum decodeLoadBalanceStrategy = config.getDecodeLoadBalanceStrategy();
        LoadBalanceStrategyEnum vitLoadBalanceStrategy = config.getVitLoadBalanceStrategy();
        this.loadBalancerMap = new EnumMap<>(RoleType.class);

        for (RoleType roleType : RoleType.values()) {
            LoadBalanceStrategyEnum strategy = roleType.getStrategy(loadBalanceStrategyByConfig, decodeLoadBalanceStrategy, vitLoadBalanceStrategy);
            loadBalancerMap.put(roleType, LoadBalanceStrategyFactory.getLoadBalancer(strategy));
        }
    }

    /**
     * Routes a request to appropriate worker nodes based on model requirements and role types.
     *
     * <p>This method implements the core routing logic for load balancing across different
     * worker types (Prefill, Decode, PDFusion, VIT).
     *
     * @param balanceContext the context containing request information, model details, and tracing span
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
        String interRequestId = balanceContext.getRequestId();
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
            recordSuccessMetrics(balanceContext, startTimeInMicros);
            response = buildSuccessResponse(interRequestId, routingResult.serverStatusList());
        } else {
            rollBackRoutingFailure(balanceContext, routingResult);
            recordFailureMetrics(balanceContext, routingResult.failedRoleType(), startTimeInMicros);
            response = buildFailureResponse(interRequestId, routingResult);
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
            recordServerSelectionMetrics(balanceContext, roleType, serverStatus);
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
            String interRequestId = balanceContext.getRequestId();

            RoleType role = serverStatus.getRole();
            LoadBalancer loadBalancer = getLoadBalancer(role);
            loadBalancer.rollBack(serverIpPort, interRequestId);
        }
    }

    /**
     * Record server selection metrics to distributed tracing span
     *
     * @param balanceContext Routing context
     * @param roleType Role type
     * @param serverStatus Selected server status
     */
    private void recordServerSelectionMetrics(BalanceContext balanceContext,
                                              RoleType roleType,
                                              ServerStatus serverStatus) {
        String rolePrefix = roleType.getCode();
        balanceContext.getSpan().setAttribute(rolePrefix + ".ip", serverStatus.getServerIp());
        balanceContext.getSpan().setAttribute(rolePrefix + ".port", String.valueOf(serverStatus.getHttpPort()));

        // For PREFILL, record prefill time
        if (roleType == RoleType.PREFILL) {
            balanceContext.getSpan().setAttribute("prefill_time", String.valueOf(serverStatus.getPrefillTime()));
        }
    }

    private void recordSuccessMetrics(BalanceContext balanceContext, long startTimeInMicros) {
        long costTimeInMicros = System.nanoTime() / 1000 - startTimeInMicros;
        balanceContext.getSpan().setAttribute("routing_duration_us", String.valueOf(costTimeInMicros));
    }

    private void recordFailureMetrics(BalanceContext balanceContext, RoleType failedRoleType, long startTimeInMicros) {
        long costTimeInMicros = System.nanoTime() / 1000 - startTimeInMicros;
        balanceContext.getSpan().setAttribute("routing_duration_us", String.valueOf(costTimeInMicros));
        balanceContext.getSpan().setAttribute("failed_role_type", failedRoleType.name());
        balanceContext.getSpan().setAttribute("error_type", failedRoleType.getErrorType().name());
    }

    private Response buildSuccessResponse(String interRequestId, List<ServerStatus> serverStatusList) {
        Response response = new Response();
        response.setInterRequestId(interRequestId);
        response.setSuccess(true);
        response.setServerStatus(serverStatusList);
        return response;
    }

    private Response buildFailureResponse(String interRequestId, RoutingResult routingResult) {
        StrategyErrorType errorType = routingResult.failedRoleType().getErrorType();
        String detailMessage = routingResult.errorMessage();

        Response response = new Response();
        response.setInterRequestId(interRequestId);
        response.setSuccess(false);
        response.setCode(errorType.getErrorCode());
        response.setErrorMessage(errorType.getErrorMsg() + ": " + detailMessage);
        return response;
    }
}
