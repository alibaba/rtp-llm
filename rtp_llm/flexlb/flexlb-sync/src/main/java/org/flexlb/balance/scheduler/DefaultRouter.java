package org.flexlb.balance.scheduler;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.DecodeEndpoint.AdmissionSnapshot;
import org.flexlb.balance.scheduler.priority.AdmissionFailureClassifier;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.policy.GroupRoutingDecision;
import org.flexlb.balance.policy.GroupRoutingPolicy;
import org.flexlb.balance.strategy.LoadBalanceStrategy;
import org.flexlb.balance.strategy.LoadBalanceStrategyFactory;
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
@DependsOn({
        "randomStrategy",
        "costBasedDecodeStrategy",
        "costBasedPrefillStrategy",
        "shortestTtftStrategy"
})
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
            LoadBalanceStrategyEnum strategy = config.strategyFor(roleType);
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
        ModelWorkerStatus workerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS;
        List<RoleType> roleTypeList = workerStatus.getRoleTypeList();
        if (CollectionUtils.isEmpty(roleTypeList)) {
            Logger.debug("No worker roles registered yet (total workers: {})", workerStatus.getWorkerTotalCount());
            return Response.error(NO_AVAILABLE_WORKER);
        }

        // 3. Execute routing decision
        RoutingResult routingResult = routeByRoleType(balanceContext, roleTypeList);

        // 4. Build response based on routing result
        if (routingResult.success()) {
            return buildSuccessResponse(routingResult.serverStatusList());
        }

        ServerStatus failure = routingResult.failure();
        try {
            return buildFailureResponse(endpointRegistry, balanceContext, failure);
        } finally {
            rollBackRoutingFailure(balanceContext, routingResult);
        }
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
    private RoutingResult routeByRoleType(BalanceContext balanceContext, List<RoleType> roleTypeList) {
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
            ServerStatus serverStatus;
            try {
                serverStatus = loadBalanceStrategy.select(balanceContext, roleType, group);
            } catch (RuntimeException failure) {
                Logger.warn("Worker selection failed for role {}", roleType, failure);
                serverStatus = ServerStatus.code(StrategyErrorType.BATCH_DISPATCH_FAILED,
                        roleType.getCode() + " selection failed: " + failure.getMessage());
            }

            if (!serverStatus.isSuccess()) {
                // Selection failed, return failure result
                Logger.warn("Failed to select {} worker: {}", roleType.getCode(), serverStatus.getMessage());
                serverStatus.setRole(roleType);
                serverStatus.setGroup(group);
                return RoutingResult.failure(serverStatusList, serverStatus);
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
            RoleType role = serverStatus.getRole();

            WorkerEndpoint ep = endpointRegistry.get(role, serverIpPort);
            if (ep == null) {
                Logger.debug("DefaultRouter.rollBack: endpoint not found for ipPort={}", serverIpPort);
                continue;
            }

            LoadBalanceStrategy loadBalanceStrategy = getLoadBalanceStrategy(role);
            loadBalanceStrategy.rollBack(ep, requestId);
        }
    }

    private Response buildSuccessResponse(List<ServerStatus> serverStatusList) {
        Response response = new Response();
        response.setSuccess(true);
        response.setServerStatus(serverStatusList);
        return response;
    }

    /** Shared by ordinary routing and Prefill placement after Decode eviction. */
    public static Response buildFailureResponse(EndpointRegistry registry, BalanceContext ctx, ServerStatus failure) {
        Response response = Response.error(failure);
        boolean classifyDecode = failure.getRole() == RoleType.DECODE
                && StrategyErrorType.fromErrorCode(failure.getCode()).isCapacityRejection();
        List<AdmissionSnapshot> decodes = new ArrayList<>();
        String group = failure.getGroup();
        RoleType role = failure.getRole();
        List<Map<String, Object>> prefills = new ArrayList<>();
        List<Map<String, Object>> decodeValues = new ArrayList<>();
        RoleType prefillRole = role == RoleType.PDFUSION ? RoleType.PDFUSION : RoleType.PREFILL;
        if (role == RoleType.PREFILL || role == RoleType.PDFUSION || role == RoleType.DECODE) {
            registry.getPrefillEndpoints(prefillRole).forEach((id, endpoint) -> {
                if (group == null || group.equals(endpoint.getStatus().getGroup())) {
                    prefills.add(Map.of("endpoint", id, "alive", endpoint.getStatus().isAlive(),
                            "queueDepth", endpoint.getBatcher().queueSize(), "pending", endpoint.realPendingCount()));
                }
            });
        }
        if (role == RoleType.PREFILL || role == RoleType.DECODE) {
            registry.getDecodeEndpoints().forEach((id, endpoint) -> {
                if (group != null && !group.equals(endpoint.getStatus().getGroup())) {
                    return;
                }
                boolean alive = endpoint.getStatus().isAlive();
                AdmissionSnapshot snapshot = classifyDecode && alive ? endpoint.admissionSnapshot() : null;
                if (snapshot != null) {
                    decodes.add(snapshot);
                }
                decodeValues.add(snapshot != null
                        ? Map.of("endpoint", id, "alive", true, "engineLoad", snapshot.engineLoad(),
                                "kvAvailable", snapshot.kvAvailable(), "kvTotal", snapshot.kvTotal(),
                                "hardKvReserved", snapshot.hardKvReserved())
                        : Map.of("endpoint", id, "alive", alive,
                                "engineLoad", endpoint.getEngineLoad(), "kvAvailable", endpoint.realKvAvailable(),
                                "kvTotal", endpoint.realKvTotal(), "hardKvReserved", endpoint.inflightHardKvReserved()));
            });
        }
        if (!decodes.isEmpty()) {
            Long limit = ctx.getConfig().getRouter().getRoles().getDecode().getAvailability().getMaxEngineRequests();
            response = AdmissionFailureClassifier.classifyDecode(ctx.getPriority(), ctx.getRequest().getSeqLen(),
                    limit == null ? 0 : limit, decodes);
            response.setFailedRole(failure.getRole());
            response.setFailedGroup(group);
        }
        ctx.setSchedulingDiagnostics(Map.of("cause", failure.getMessage() == null ? "worker selection failed" : failure.getMessage(),
                "capturedAtMs", System.currentTimeMillis(), "role", role == null ? "" : role.getCode(),
                "group", group == null ? "" : group, "prefill", List.copyOf(prefills), "decode", List.copyOf(decodeValues)));
        return response;
    }

}
