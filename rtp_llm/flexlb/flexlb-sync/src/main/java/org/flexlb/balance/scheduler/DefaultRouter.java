package org.flexlb.balance.scheduler;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.policy.GroupRoutingPolicy;
import org.flexlb.balance.strategy.BatchLoadBalancer;
import org.flexlb.balance.strategy.LoadBalanceStrategy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.BatchScheduleRequest;
import org.flexlb.dao.loadbalance.BatchScheduleResponse;
import org.flexlb.dao.loadbalance.BatchScheduleTarget;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.RoutingResult;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.EngineType;
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
@DependsOn({
        "randomStrategy",
        "costBasedDecodeStrategy",
        "costBasedPrefillStrategy",
        "shortestTtftStrategy",
        "roundRobinStrategy"
})
public class DefaultRouter implements Router {

    private final Map<RoleType, LoadBalanceStrategy> loadBalanceStrategyMap;
    private final GroupRoutingPolicy groupRoutingPolicy;
    private final EndpointRegistry endpointRegistry;
    private final LoadBalanceStrategy batchLoadBalanceStrategy;
    private final LoadBalanceStrategyEnum batchStrategyType;
    private final int batchScheduleMaxCount;
    private final ModelMetaConfig modelMetaConfig;
    private final boolean embeddingEngine;

    public DefaultRouter(
            ConfigService configService,
            GroupRoutingPolicy groupRoutingPolicy,
            EndpointRegistry endpointRegistry,
            ModelMetaConfig modelMetaConfig) {
        this.groupRoutingPolicy = groupRoutingPolicy;
        this.endpointRegistry = endpointRegistry;
        this.modelMetaConfig = modelMetaConfig;

        FlexlbConfig config = configService.loadBalanceConfig();
        this.embeddingEngine = config.getEngineType() == EngineType.EMBEDDING;
        this.loadBalanceStrategyMap = new EnumMap<>(RoleType.class);
        for (RoleType roleType : RoleType.values()) {
            LoadBalanceStrategyEnum strategy = config.getStrategyForRoleType(roleType);
            if (strategy != null) {
                loadBalanceStrategyMap.put(
                        roleType,
                        LoadBalanceStrategyFactory.getLoadBalanceStrategy(strategy));
                Logger.info("DefaultRouter role={}: schedule={}", roleType, strategy);
            }
        }

        this.batchStrategyType = config.getBatchLoadBalanceStrategy();
        this.batchLoadBalanceStrategy =
                LoadBalanceStrategyFactory.getLoadBalanceStrategy(batchStrategyType);
        this.batchScheduleMaxCount = config.getBatchScheduleMaxCount();
        if (batchScheduleMaxCount < 1) {
            throw new IllegalStateException(
                    "batchScheduleMaxCount must be >= 1, got " + batchScheduleMaxCount
                            + "; check BATCH_SCHEDULE_MAX_COUNT");
        }
        Logger.info("DefaultRouter batchSchedule={}, batchScheduleMaxCount={}",
                batchStrategyType, batchScheduleMaxCount);
    }

    @Override
    public Response route(BalanceContext context) {
        Response validationResponse = validateRequest(context);
        if (validationResponse != null) {
            return validationResponse;
        }
        if (embeddingEngine) {
            return Response.error(
                    StrategyErrorType.INVALID_REQUEST,
                    "engineType=EMBEDDING is batch-only; use /batch_schedule because "
                            + "the single-request response cannot express arpc_port");
        }

        ModelWorkerStatus workerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        List<RoleType> roleTypes = workerStatus.getRoleTypeList();
        if (CollectionUtils.isEmpty(roleTypes)) {
            Logger.debug("No worker roles registered yet (total workers: {})",
                    workerStatus.getWorkerTotalCount());
            return Response.error(NO_AVAILABLE_WORKER, noWorkerDetail());
        }

        RoutingResult routingResult = routeByRoleType(context, roleTypes);
        if (routingResult.success()) {
            return buildSuccessResponse(routingResult.serverStatusList());
        }

        rollBackRoutingFailure(context, routingResult);
        return buildFailureResponse(routingResult);
    }

    @Override
    public BatchScheduleResponse batchSchedule(BatchScheduleRequest request) {
        if (request == null) {
            return BatchScheduleResponse.error(
                    StrategyErrorType.INVALID_REQUEST, "batch_schedule request is null");
        }
        int count = request.getBatchCount();
        if (count < 1 || count > batchScheduleMaxCount) {
            return BatchScheduleResponse.error(
                    StrategyErrorType.INVALID_REQUEST,
                    "batch_count must be in [1, " + batchScheduleMaxCount + "]");
        }
        if (!request.isAssignBe() && !request.isAssignFe()) {
            return BatchScheduleResponse.error(
                    StrategyErrorType.INVALID_REQUEST,
                    "batch_schedule must request at least one of assign_be or assign_fe");
        }

        if (!request.isAssignBe()) {
            // FE-only allocation deliberately bypasses worker topology and the batch strategy. The
            // outer master handler stamps fe_url onto these index-preserving placeholders. Keeping
            // this branch ahead of role validation also lets healthy FE fanout operate while the BE
            // route table is warming up or the deployment is multi-role.
            List<BatchScheduleTarget> targets = new ArrayList<>(count);
            for (int i = 0; i < count; i++) {
                targets.add(new BatchScheduleTarget());
            }
            return BatchScheduleResponse.success(targets);
        }

        List<RoleType> configuredRoles = modelMetaConfig.getConfiguredRoleTypes();
        if (configuredRoles.size() != 1) {
            return rejectRoleTopology("Configured", configuredRoles);
        }

        List<RoleType> runtimeRoles =
                EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleTypeList();
        if (CollectionUtils.isEmpty(runtimeRoles)) {
            return BatchScheduleResponse.error(NO_AVAILABLE_WORKER, noWorkerDetail());
        }
        if (runtimeRoles.size() != 1) {
            return rejectRoleTopology("Detected", runtimeRoles);
        }

        RoleType roleType = runtimeRoles.get(0);
        if (configuredRoles.get(0) != roleType) {
            return BatchScheduleResponse.error(
                    NO_AVAILABLE_WORKER,
                    "configured role " + configuredRoles.get(0)
                            + " does not match runtime role " + roleType);
        }

        if (!(batchLoadBalanceStrategy instanceof BatchLoadBalancer batchLoadBalancer)) {
            return BatchScheduleResponse.error(
                    StrategyErrorType.INVALID_REQUEST,
                    "batch strategy " + batchStrategyType
                            + " does not support batch_schedule; check "
                            + "BATCH_LOAD_BALANCE_STRATEGY");
        }

        List<BatchScheduleTarget> targets =
                batchLoadBalancer.selectBatch(count, roleType, null);
        if (targets == null || targets.isEmpty()) {
            return BatchScheduleResponse.error(roleType.getErrorType());
        }
        if (targets.size() != count) {
            return BatchScheduleResponse.error(
                    StrategyErrorType.INVALID_REQUEST,
                    "batch strategy " + batchStrategyType + " returned "
                            + targets.size() + " targets for batch_count " + count);
        }
        return BatchScheduleResponse.success(targets);
    }

    /** Distinguish a missing route table from discovery resolving a configured route to no hosts. */
    private String noWorkerDetail() {
        List<String> addresses = modelMetaConfig.getConfiguredDiscoveryAddresses();
        String detail = CollectionUtils.isEmpty(addresses)
                ? "master not ready: no service route registered; check MODEL_SERVICE_CONFIG is present and parses"
                : "master not ready: route table is loaded but no worker was discovered through " + addresses
                        + "; check the configured discovery service and its host membership";
        return NO_AVAILABLE_WORKER.getErrorMsg() + ": " + detail;
    }

    private static BatchScheduleResponse rejectRoleTopology(
            String source, List<RoleType> roles) {
        return BatchScheduleResponse.error(
                StrategyErrorType.INVALID_REQUEST,
                "batch_schedule supports single-role deployments only and requires "
                        + "exactly one configured and one runtime role; use /schedule "
                        + "for multi-role routing. " + source + " roles: " + roles);
    }

    private Response validateRequest(BalanceContext context) {
        if (context.getRequest() == null) {
            Logger.error("masterRequest is null");
            return Response.error(StrategyErrorType.INVALID_REQUEST);
        }
        return null;
    }

    private RoutingResult routeByRoleType(
            BalanceContext context, List<RoleType> roleTypes) {
        List<ServerStatus> serverStatusList = new ArrayList<>();
        GroupRoutingDecision groupDecision = groupRoutingPolicy.route(context);
        String policyGroup = groupDecision.group();
        String group = policyGroup;
        if (groupDecision.hasGroup()) {
            Logger.info(
                    "Group routing policy selected group, requestId: {}, policy: {}, group: {}",
                    context.getRequestId(), groupDecision.policyName(), group);
        }

        for (RoleType roleType : roleTypes) {
            LoadBalanceStrategy strategy = getLoadBalanceStrategy(roleType);
            if (strategy == null) {
                return RoutingResult.failure(
                        serverStatusList,
                        roleType,
                        "no load-balancing strategy configured");
            }
            ServerStatus serverStatus = strategy.select(context, roleType, group);
            if (!serverStatus.isSuccess()) {
                Logger.warn("Failed to select {} worker: {}",
                        roleType.getCode(), serverStatus.getMessage());
                return RoutingResult.failure(
                        serverStatusList, roleType, serverStatus.getMessage());
            }

            serverStatusList.add(serverStatus);
            if (StringUtils.isBlank(policyGroup)) {
                group = serverStatus.getGroup();
            }
        }
        return RoutingResult.success(serverStatusList);
    }

    private LoadBalanceStrategy getLoadBalanceStrategy(RoleType roleType) {
        return loadBalanceStrategyMap.get(roleType);
    }

    private void rollBackRoutingFailure(
            BalanceContext context, RoutingResult routingResult) {
        for (ServerStatus serverStatus : routingResult.serverStatusList()) {
            String serverIpPort =
                    serverStatus.getServerIp() + ":" + serverStatus.getHttpPort();
            RoleType role = serverStatus.getRole();
            WorkerEndpoint endpoint = endpointRegistry.get(role, serverIpPort);
            if (endpoint == null) {
                Logger.debug("DefaultRouter.rollBack: endpoint not found for ipPort={}",
                        serverIpPort);
                continue;
            }
            LoadBalanceStrategy strategy = getLoadBalanceStrategy(role);
            strategy.rollBack(endpoint, context.getRequestId());
        }
    }

    private Response buildSuccessResponse(List<ServerStatus> serverStatusList) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(serverStatusList);
        return response;
    }

    private Response buildFailureResponse(RoutingResult routingResult) {
        StrategyErrorType errorType = routingResult.failedRoleType().getErrorType();
        Response response = new Response();
        response.setSuccess(false);
        response.setCode(errorType.getErrorCode());
        response.setErrorMessage(
                errorType.buildErrorMessage(routingResult.errorMessage()));
        return response;
    }
}
