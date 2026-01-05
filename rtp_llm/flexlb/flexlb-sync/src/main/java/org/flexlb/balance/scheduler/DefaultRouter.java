package org.flexlb.balance.scheduler;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.MapUtils;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.MasterRequest;
import org.flexlb.dao.loadbalance.MasterResponse;
import org.flexlb.dao.loadbalance.RoutingResult;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.sync.status.ModelWorkerStatus;
import org.flexlb.util.LoggingUtils;
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
        LoadBalanceStrategyEnum loadBalanceStrategyByConfig = configService.loadBalanceConfig().getLoadBalanceStrategy();
        this.loadBalancerMap = new EnumMap<>(RoleType.class);

        for (RoleType roleType : RoleType.values()) {
            LoadBalanceStrategyEnum strategy = roleType.getStrategy(loadBalanceStrategyByConfig);
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
     * @return MasterResponse containing selected server statuses or error information
     */
    @Override
    public MasterResponse route(BalanceContext balanceContext) {
        long startTimeInMicros = System.nanoTime() / 1000;

        // 1. 验证请求
        MasterResponse validationResponse = validateRequest(balanceContext);
        if (validationResponse != null) {
            return validationResponse;
        }

        // 2. 获取路由配置
        MasterRequest masterRequest = balanceContext.getMasterRequest();
        String modelName = masterRequest.getModel();
        String interRequestId = balanceContext.getInterRequestId();
        ModelWorkerStatus workerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP.get(modelName);
        List<RoleType> roleTypeList = workerStatus.getRoleTypeList();
        if (CollectionUtils.isEmpty(roleTypeList)) {
            return MasterResponse.error(NO_AVAILABLE_WORKER);
        }

        // 3. 执行路由决策
        RoutingResult routingResult = routeByRoleType(balanceContext, roleTypeList);

        // 4. 根据路由结果构建响应
        MasterResponse masterResponse;
        if (routingResult.success()) {
            recordSuccessMetrics(balanceContext, startTimeInMicros);
            masterResponse = buildSuccessResponse(interRequestId, routingResult.serverStatusList());
        } else {
            rollBackRoutingFailure(balanceContext, routingResult);
            recordFailureMetrics(balanceContext, routingResult.failedRoleType(), startTimeInMicros);
            masterResponse = buildFailureResponse(interRequestId, routingResult);
        }

        return masterResponse;
    }

    /**
     * Validates the incoming request and checks model availability.
     *
     * @param balanceContext the context to validate
     * @return error response if validation fails, null if validation succeeds
     */
    private MasterResponse validateRequest(BalanceContext balanceContext) {
        if (balanceContext.getMasterRequest() == null) {
            LoggingUtils.error("masterRequest is null");
            return MasterResponse.error(StrategyErrorType.INVALID_REQUEST);
        }

        String modelName = balanceContext.getMasterRequest().getModel();
        Map<String, ModelWorkerStatus> workerStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS_MAP;

        if (MapUtils.isEmpty(workerStatusMap)) {
            LoggingUtils.error("targetModelRoleWorkerStatusMap is empty");
            return MasterResponse.error(NO_AVAILABLE_WORKER);
        }

        if (!workerStatusMap.containsKey(modelName)) {
            LoggingUtils.error("targetModelRoleWorkerStatusMap has no key named {}", modelName);
            return MasterResponse.error(NO_AVAILABLE_WORKER);
        }

        return null;
    }

    /**
     * 执行路由决策，为每个角色类型选择最优服务器
     *
     * @param balanceContext 路由上下文
     * @param roleTypeList 需要的角色类型列表
     * @return 路由结果
     */
    public RoutingResult routeByRoleType(BalanceContext balanceContext, List<RoleType> roleTypeList) {
        List<ServerStatus> serverStatusList = new ArrayList<>();
        String group = null;

        for (RoleType roleType : roleTypeList) {
            LoadBalancer loadBalancer = getLoadBalancer(roleType);
            ServerStatus serverStatus = loadBalancer.select(balanceContext, roleType, group);

            if (!serverStatus.isSuccess()) {
                // 选择失败，返回失败结果
                LoggingUtils.warn("Failed to select {} worker: {}", roleType.getCode(), serverStatus.getMessage());
                return RoutingResult.failure(serverStatusList, roleType, serverStatus.getMessage());
            }

            // 记录服务器选择指标
            recordServerSelectionMetrics(balanceContext, roleType, serverStatus);
            serverStatusList.add(serverStatus);

            // 更新 group，用于后续角色的亲和性选择
            group = serverStatus.getGroup();
        }

        return RoutingResult.success(serverStatusList);
    }

    /**
     * 根据角色类型获取对应的 LoadBalancer
     */
    private LoadBalancer getLoadBalancer(RoleType roleType) {
        return loadBalancerMap.get(roleType);
    }

    /**
     * 回滚处理路由失败情况
     * 如果部分Role已经选择成功但后续Role失败，需要回滚之前选择Role的本地增量更新
     *
     * @param balanceContext 路由上下文
     * @param routingResult 路由结果
     */
    private void rollBackRoutingFailure(BalanceContext balanceContext, RoutingResult routingResult) {

        List<ServerStatus> partialResults = routingResult.serverStatusList();
        for (ServerStatus serverStatus : partialResults) {
            String modelName = balanceContext.getMasterRequest().getModel();
            String serverIpPort = serverStatus.getServerIp() + ":" + serverStatus.getHttpPort();
            String interRequestId = balanceContext.getInterRequestId();

            RoleType role = serverStatus.getRole();
            LoadBalancer loadBalancer = getLoadBalancer(role);
            loadBalancer.rollBack(modelName, serverIpPort, interRequestId);
        }
    }

    /**
     * 记录服务器选择指标到分布式追踪 span
     *
     * @param balanceContext 路由上下文
     * @param roleType 角色类型
     * @param serverStatus 选择的服务器状态
     */
    private void recordServerSelectionMetrics(BalanceContext balanceContext,
                                              RoleType roleType,
                                              ServerStatus serverStatus) {
        String rolePrefix = roleType.getCode();
        balanceContext.getSpan().setAttribute(rolePrefix + ".ip", serverStatus.getServerIp());
        balanceContext.getSpan().setAttribute(rolePrefix + ".port", String.valueOf(serverStatus.getHttpPort()));

        // 对于 PREFILL，记录预填充时间
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

    private MasterResponse buildSuccessResponse(String interRequestId, List<ServerStatus> serverStatusList) {
        MasterResponse response = new MasterResponse();
        response.setInterRequestId(interRequestId);
        response.setSuccess(true);
        response.setServerStatus(serverStatusList);
        return response;
    }

    private MasterResponse buildFailureResponse(String interRequestId, RoutingResult routingResult) {
        StrategyErrorType errorType = routingResult.failedRoleType().getErrorType();
        String detailMessage = routingResult.errorMessage();

        MasterResponse response = new MasterResponse();
        response.setInterRequestId(interRequestId);
        response.setSuccess(false);
        response.setCode(errorType.getErrorCode());
        response.setErrorMessage(errorType.getErrorMsg() + ": " + detailMessage);
        return response;
    }
}
