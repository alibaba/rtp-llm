package org.flexlb.balance.scheduler;

import org.apache.commons.collections4.CollectionUtils;
import org.flexlb.balance.strategy.BatchLoadBalancer;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.RoutingResult;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
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
@DependsOn({"randomStrategy", "weightedCacheStrategy", "shortestTTFTStrategy", "roundRobinStrategy"})
public class DefaultRouter implements Router {

    private final Map<RoleType, LoadBalancer> loadBalancerMap;
    /**
     * Balancer for {@code /batch_schedule}, separate from {@link #loadBalancerMap} which
     * governs {@code /schedule}. Decoupling lets operators keep e.g. SHORTEST_TTFT for
     * single-request routing while the batch endpoint defaults to ROUND_ROBIN — the only
     * batch-capable strategy today and the source of batch_schedule's atomic-distribution
     * guarantee. See {@link FlexlbConfig#getBatchLoadBalanceStrategy}.
     */
    private final LoadBalancer batchLoadBalancer;
    private final int batchScheduleMaxCount;

    public DefaultRouter(ConfigService configService) {
        FlexlbConfig config = configService.loadBalanceConfig();
        this.loadBalancerMap = new EnumMap<>(RoleType.class);

        for (RoleType roleType : RoleType.values()) {
            LoadBalanceStrategyEnum strategy = config.getStrategyForRoleType(roleType);
            loadBalancerMap.put(roleType, LoadBalanceStrategyFactory.getLoadBalancer(strategy));
            Logger.warn("DefaultRouter role={}: schedule={}", roleType, strategy);
        }

        LoadBalanceStrategyEnum batchStrategy = config.getBatchLoadBalanceStrategy();
        this.batchLoadBalancer = LoadBalanceStrategyFactory.getLoadBalancer(batchStrategy);
        this.batchScheduleMaxCount = config.getBatchScheduleMaxCount();
        Logger.warn("DefaultRouter batchSchedule={}, batchScheduleMaxCount={}",
                batchStrategy, batchScheduleMaxCount);
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
     * Single-role batch dispatch. Scope: only callable when the cluster has exactly one
     * registered role. Multi-stage deployments (disaggregated PD / VL) should fan out
     * via {@link #route} per request.
     */
    public BatchScheduleResponse batchSchedule(BatchScheduleRequest batchRequest) {
        // (1) batch_count validation
        if (batchRequest == null) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST, "batch_schedule request is null");
        }
        int count = batchRequest.getBatchCount();
        if (count < 1 || count > batchScheduleMaxCount) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batch_count must be in [1, " + batchScheduleMaxCount + "]");
        }

        // (1a) sub_requests length consistency (when caller opts in to forward-compatible payload)
        List<Request> subs = batchRequest.getSubRequests();
        if (subs != null && subs.size() != count) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "sub_requests length " + subs.size() + " != batch_count " + count);
        }

        // (2) master readiness
        ModelWorkerStatus workerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        if (workerStatus == null) {
            return BatchScheduleResponse.error(NO_AVAILABLE_WORKER,
                    "master not ready or MODEL_SERVICE_CONFIG missing");
        }

        // (3) Role inference: reuse the same data source /schedule uses
        List<RoleType> roleTypes = workerStatus.getRoleTypeList();
        if (CollectionUtils.isEmpty(roleTypes)) {
            return BatchScheduleResponse.error(NO_AVAILABLE_WORKER,
                    "master not ready or MODEL_SERVICE_CONFIG missing");
        }
        if (roleTypes.size() > 1) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batch_schedule only supports single-role deployments; "
                    + "multi-stage deployments (disaggregated PD / VL) should use /schedule per request. "
                    + "Detected roles: " + roleTypes);
        }
        RoleType roleType = roleTypes.get(0);

        // (4) Batch strategy (independent of /schedule's strategy) must support batch.
        //     Default is ROUND_ROBIN; an operator-configured non-batch-capable batchStrategy
        //     fails loudly here rather than silently falling back, so misconfiguration is loud.
        if (!(this.batchLoadBalancer instanceof BatchLoadBalancer batchLoadBalancer)) {
            return BatchScheduleResponse.error(StrategyErrorType.INVALID_REQUEST,
                    "batchStrategy for role " + roleType.getCode() + " does not support batch_schedule");
        }

        // (5) RR pick N targets
        List<BatchScheduleTarget> targets = batchLoadBalancer.selectBatch(count, roleType, null);
        if (targets == null || targets.isEmpty()) {
            return BatchScheduleResponse.error(roleType.getErrorType());
        }

        return BatchScheduleResponse.success(targets);
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
}
