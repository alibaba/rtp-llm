package org.flexlb.balance.scheduler;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.policy.GroupRoutingPolicy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalanceStrategy;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
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
@DependsOn({"randomStrategy", "costBasedDecodeStrategy", "costBasedPrefillStrategy", "shortestTtftStrategy"})
public class DefaultRouter implements Router {

    private final Map<RoleType, LoadBalanceStrategy> loadBalanceStrategyMap;
    private final GroupRoutingPolicy groupRoutingPolicy;
    private final EndpointRegistry endpointRegistry;

    public DefaultRouter(ConfigService configService, GroupRoutingPolicy groupRoutingPolicy,
                         EndpointRegistry endpointRegistry) {
        this.groupRoutingPolicy = groupRoutingPolicy;
        this.endpointRegistry = endpointRegistry;
        FlexlbConfig config = configService.loadBalanceConfig();
        this.loadBalanceStrategyMap = new EnumMap<>(RoleType.class);

        for (RoleType roleType : RoleType.values()) {
            LoadBalanceStrategyEnum strategy = config.getStrategyForRoleType(roleType);
            if (strategy != null) {
                loadBalanceStrategyMap.put(roleType, LoadBalanceStrategyFactory.getLoadBalanceStrategy(strategy));
            }
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
            Logger.warn("No worker roles registered yet (total workers: {})", workerStatus.getWorkerTotalCount());
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
        GroupRoutingDecision groupRoutingDecision = groupRoutingPolicy.route(balanceContext);
        String policyGroup = groupRoutingDecision.group();
        String group = policyGroup;
        if (groupRoutingDecision.hasGroup()) {
            Logger.info("Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    balanceContext.getRequestId(), groupRoutingDecision.policyName(), group);
        }

        for (RoleType roleType : roleTypeList) {
            LoadBalanceStrategy loadBalanceStrategy = getLoadBalanceStrategy(roleType);
            ServerStatus serverStatus = loadBalanceStrategy.select(balanceContext, roleType, group);

            if (!serverStatus.isSuccess()) {
                // Selection failed, return failure result
                Logger.warn("Failed to select {} worker: {}", roleType.getCode(), serverStatus.getMessage());
                return RoutingResult.failure(serverStatusList, roleType, serverStatus.getMessage());
            }

            // Record server selection metrics
            serverStatusList.add(serverStatus);

            // Update group for affinity-based selection of subsequent roles
            if (StringUtils.isBlank(policyGroup)) {
                group = serverStatus.getGroup();
            }
        }

        return RoutingResult.success(serverStatusList);
    }

    /**
     * Get LoadBalanceStrategy based on role type
     */
    private LoadBalanceStrategy getLoadBalanceStrategy(RoleType roleType) {
        return loadBalanceStrategyMap.get(roleType);
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

            WorkerEndpoint ep = endpointRegistry.get(serverIpPort);
            if (ep == null) {
                Logger.warn("DefaultRouter.rollBack: endpoint not found for ipPort={}", serverIpPort);
                continue;
            }

            RoleType role = serverStatus.getRole();
            LoadBalanceStrategy loadBalanceStrategy = getLoadBalanceStrategy(role);
            loadBalanceStrategy.rollBack(ep, requestId);
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
}
