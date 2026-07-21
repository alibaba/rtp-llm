package org.flexlb.engine.grpc.client;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.discovery.RoutingServiceDiscovery;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.kvcm.grpc.QueryType;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Resolves the KVCM query type from the engine's KV-cache group metadata.
 */
@Slf4j
@Component
public class KvcmQueryTypeResolver {

    private static final String FULL_ATTENTION_KIND = "full_attention";
    private static final String MAMBA_ATTENTION_KIND = "mamba";
    private static final QueryType DEFAULT_QUERY_TYPE = QueryType.QT_PREFIX_MATCH;

    private final ServiceRoute serviceRoute;
    private final RoutingServiceDiscovery serviceDiscovery;
    private final EngineGrpcClient engineGrpcClient;
    private final long requestTimeoutMs;
    private final ConcurrentMap<String, ConcurrentMap<RoleType, QueryType>> queryTypeByGroupAndRole =
            new ConcurrentHashMap<>();

    public KvcmQueryTypeResolver(
            ModelMetaConfig modelMetaConfig,
            RoutingServiceDiscovery serviceDiscovery,
            EngineGrpcClient engineGrpcClient) {
        this.serviceRoute = modelMetaConfig.getServiceRoutes().stream().findFirst().orElse(null);
        this.serviceDiscovery = serviceDiscovery;
        this.engineGrpcClient = engineGrpcClient;
        this.requestTimeoutMs = serviceRoute != null && serviceRoute.getKvcm() != null
                ? serviceRoute.getKvcm().getRequestTimeoutMs()
                : KvcmConfig.DEFAULT_REQUEST_TIMEOUT_MS;
    }

    public QueryType resolve(RoleType roleType, String group) {
        ConcurrentMap<RoleType, QueryType> queryTypeByRole =
                queryTypeByGroupAndRole.get(StringUtils.defaultString(group));
        return queryTypeByRole == null
                ? DEFAULT_QUERY_TYPE
                : queryTypeByRole.getOrDefault(roleType, DEFAULT_QUERY_TYPE);
    }

    void refresh() {
        if (serviceRoute == null) {
            return;
        }

        Map<String, QueryType> queryTypeByEndpointAddress = new HashMap<>();
        Set<String> attemptedEndpointAddresses = new HashSet<>();
        for (GroupRoleEndPoint roleEndpoints : serviceRoute.getRoleEndpoints()) {
            String group = StringUtils.defaultString(roleEndpoints.getGroup());
            for (RoleType roleType : RoleType.values()) {
                Endpoint endpoint = roleEndpoints.getRoleEndpoint(roleType);
                if (endpoint == null) {
                    continue;
                }

                String endpointAddress = endpoint.getAddress();
                if (attemptedEndpointAddresses.add(endpointAddress)) {
                    QueryType resolvedQueryType = queryAvailableReplica(endpoint);
                    if (resolvedQueryType != null) {
                        queryTypeByEndpointAddress.put(endpointAddress, resolvedQueryType);
                    }
                }
                QueryType queryType = queryTypeByEndpointAddress.get(endpointAddress);
                if (queryType == null) {
                    continue;
                }
                ConcurrentMap<RoleType, QueryType> queryTypeByRole =
                        queryTypeByGroupAndRole.computeIfAbsent(group, ignored -> new ConcurrentHashMap<>());
                QueryType previous = queryTypeByRole.get(roleType);
                if (queryType != previous) {
                    queryTypeByRole.put(roleType, queryType);
                    log.info("Updated KVCM query type, group={}, role={}, endpoint={}, queryType={}",
                            group, roleType, endpointAddress, queryType);
                }
            }
        }
    }

    private QueryType queryAvailableReplica(Endpoint endpoint) {
        List<WorkerHost> hosts = serviceDiscovery.getHosts(endpoint);
        for (WorkerHost host : hosts) {
            QueryType queryType = queryEngine(host);
            if (queryType != null) {
                return queryType;
            }
        }
        return null;
    }

    private QueryType queryEngine(WorkerHost host) {
        try {
            EngineRpcService.KvCacheGroupListPB response = engineGrpcClient.getKvCacheGroupsMetadata(
                    host.getIp(),
                    host.getWorkerStatusPort(),
                    EngineRpcService.KvCacheGroupsRequestPB.getDefaultInstance(),
                    requestTimeoutMs);
            if (response.getErrCode()
                    != EngineRpcService.KvCacheGroupMetadataErrorCode.KV_CACHE_GROUP_METADATA_OK) {
                log.debug("KV-cache group metadata is unavailable for {}:{}, code={}, message={}",
                        host.getIp(), host.getWorkerStatusPort(), response.getErrCode(), response.getErrMsg());
                return null;
            }
            if (response.getItemsCount() == 0) {
                log.debug("KV-cache group metadata is empty for {}:{}",
                        host.getIp(), host.getWorkerStatusPort());
                return null;
            }

            boolean containsMambaAttention = response.getItemsList().stream()
                    .anyMatch(item -> MAMBA_ATTENTION_KIND.equalsIgnoreCase(item.getKind()));
            if (containsMambaAttention) {
                return QueryType.QT_PREFIX_MATCH_WITH_MAMBA;
            }
            boolean containsOnlyFullAttention = response.getItemsList().stream()
                    .allMatch(item -> FULL_ATTENTION_KIND.equalsIgnoreCase(item.getKind()));
            if (containsOnlyFullAttention) {
                return QueryType.QT_PREFIX_MATCH;
            }
            log.debug("KV-cache group metadata contains unsupported attention kinds for {}:{}: {}",
                    host.getIp(), host.getWorkerStatusPort(), response.getItemsList());
            return null;
        } catch (Exception e) {
            log.debug("Failed to query KV-cache group metadata from {}:{}",
                    host.getIp(), host.getWorkerStatusPort(), e);
            return null;
        }
    }
}
