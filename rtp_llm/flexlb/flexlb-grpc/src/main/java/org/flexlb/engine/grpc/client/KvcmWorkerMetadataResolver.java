package org.flexlb.engine.grpc.client;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.kvcm.grpc.QueryType;
import org.springframework.stereotype.Component;

import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

/**
 * Resolves KVCM request metadata from one traversal of in-memory worker status.
 */
@Slf4j
@Component
public class KvcmWorkerMetadataResolver {

    private static final QueryType DEFAULT_QUERY_TYPE = QueryType.QT_PREFIX_MATCH;

    private final ServiceRoute serviceRoute;
    private final String configuredNamespace;
    private final WorkerStatusProvider workerStatusProvider;
    private final ConcurrentMap<String, ConcurrentMap<RoleType, String>> namespaceByGroupAndRole =
            new ConcurrentHashMap<>();
    private final ConcurrentMap<String, ConcurrentMap<RoleType, QueryType>> queryTypeByGroupAndRole =
            new ConcurrentHashMap<>();
    private final ConcurrentMap<RoleType, QueryType> queryTypeByRoleForCrossGroupRouting =
            new ConcurrentHashMap<>();
    private final ConcurrentMap<String, ConcurrentMap<RoleType, Integer>> rollbackBlocksByGroupAndRole =
            new ConcurrentHashMap<>();
    private final ConcurrentMap<RoleType, Integer> rollbackBlocksByRoleForCrossGroupRouting =
            new ConcurrentHashMap<>();

    public KvcmWorkerMetadataResolver(CacheMatchConfiguration configuration, WorkerStatusProvider workerStatusProvider) {
        this.serviceRoute = configuration.getKvcmServiceRoute();
        KvcmConfig config = configuration.getKvcmConfig();
        this.configuredNamespace = configuration.isKvcmEnabled()
                ? StringUtils.trimToNull(config.getNamespace())
                : null;
        this.workerStatusProvider = workerStatusProvider;
    }

    public String resolveNamespace(RoleType roleType, String group, long blockSize) {
        String namespace = configuredNamespace;
        if (namespace == null) {
            ConcurrentMap<RoleType, String> namespaceByRole =
                    namespaceByGroupAndRole.get(StringUtils.defaultString(group));
            namespace = namespaceByRole == null ? null : namespaceByRole.get(roleType);
        }
        return namespace == null ? null : namespace + "_" + blockSize;
    }

    public QueryType resolveQueryType(RoleType roleType, String group) {
        if (StringUtils.isBlank(group)) {
            return queryTypeByRoleForCrossGroupRouting.get(roleType);
        }
        ConcurrentMap<RoleType, QueryType> queryTypeByRole = queryTypeByGroupAndRole.get(group);
        return queryTypeByRole == null ? null : queryTypeByRole.get(roleType);
    }

    public int resolveCacheMatchRollbackBlocks(RoleType roleType, String group) {
        if (StringUtils.isBlank(group)) {
            return rollbackBlocksByRoleForCrossGroupRouting.getOrDefault(roleType, 0);
        }
        ConcurrentMap<RoleType, Integer> rollbackBlocksByRole = rollbackBlocksByGroupAndRole.get(group);
        return rollbackBlocksByRole == null ? 0 : rollbackBlocksByRole.getOrDefault(roleType, 0);
    }

    public boolean usesConfiguredNamespace() {
        return configuredNamespace != null;
    }

    public void refreshNamespacesAndQueryTypes() {
        if (serviceRoute == null) {
            return;
        }

        boolean namespaceChanged = false;
        boolean queryTypeChanged = false;
        boolean rollbackBlocksChanged = false;
        for (GroupRoleEndPoint roleEndpoints : serviceRoute.getRoleEndpoints()) {
            String group = StringUtils.defaultString(roleEndpoints.getGroup());
            for (RoleType roleType : RoleType.values()) {
                Endpoint endpoint = roleEndpoints.getRoleEndpoint(roleType);
                if (endpoint == null) {
                    continue;
                }

                ResolvedWorkerMetadata metadata =
                        resolveWorkerMetadata(workerStatusProvider.getWorkerStatuses(roleType, group));
                if (configuredNamespace == null && metadata.namespace() != null) {
                    ConcurrentMap<RoleType, String> namespaceByRole =
                            namespaceByGroupAndRole.computeIfAbsent(group, ignored -> new ConcurrentHashMap<>());
                    if (!metadata.namespace().equals(namespaceByRole.get(roleType))) {
                        namespaceByRole.put(roleType, metadata.namespace());
                        namespaceChanged = true;
                    }
                }
                if (metadata.queryType() != null) {
                    ConcurrentMap<RoleType, QueryType> queryTypeByRole =
                            queryTypeByGroupAndRole.computeIfAbsent(group, ignored -> new ConcurrentHashMap<>());
                    if (metadata.queryType() != queryTypeByRole.get(roleType)) {
                        queryTypeByRole.put(roleType, metadata.queryType());
                        queryTypeChanged = true;
                    }
                }
                if (metadata.cacheMatchRollbackBlocks() != null) {
                    ConcurrentMap<RoleType, Integer> rollbackBlocksByRole = rollbackBlocksByGroupAndRole.computeIfAbsent(
                            group, ignored -> new ConcurrentHashMap<>());
                    if (!metadata.cacheMatchRollbackBlocks().equals(rollbackBlocksByRole.get(roleType))) {
                        rollbackBlocksByRole.put(roleType, metadata.cacheMatchRollbackBlocks());
                        rollbackBlocksChanged = true;
                    }
                }
            }
        }

        if (queryTypeChanged) {
            refreshCrossGroupQueryTypes();
            log.info("Updated KVCM role query types: {}", queryTypeByGroupAndRole);
        }
        if (rollbackBlocksChanged) {
            refreshCrossGroupRollbackBlocks();
            log.info("Updated KVCM cache match rollback blocks: {}", rollbackBlocksByGroupAndRole);
        }
        if (namespaceChanged) {
            log.info("Updated KVCM role namespaces: {}", namespaceByGroupAndRole);
        }
    }

    private ResolvedWorkerMetadata resolveWorkerMetadata(Collection<WorkerStatus> workerStatuses) {
        String namespace = null;
        QueryType queryType = null;
        Integer cacheMatchRollbackBlocks = null;
        if (workerStatuses == null || workerStatuses.isEmpty()) {
            return new ResolvedWorkerMetadata(null, null, null);
        }
        for (WorkerStatus workerStatus : workerStatuses) {
            if (workerStatus == null) {
                continue;
            }
            if (configuredNamespace == null && namespace == null
                    && StringUtils.isNotBlank(workerStatus.getDeploymentName())) {
                namespace = workerStatus.getDeploymentName();
            }
            if (queryType == null) {
                queryType = toQueryType(workerStatus.getKvCacheGroupMode());
            }
            if (cacheMatchRollbackBlocks == null) {
                cacheMatchRollbackBlocks = workerStatus.getCacheMatchRollbackBlocks();
            }
            if ((configuredNamespace != null || namespace != null)
                    && queryType != null
                    && cacheMatchRollbackBlocks != null) {
                break;
            }
        }
        return new ResolvedWorkerMetadata(namespace, queryType, cacheMatchRollbackBlocks);
    }

    private QueryType toQueryType(KvCacheGroupMode mode) {
        if (mode == null) {
            return null;
        }
        return switch (mode) {
            case FULL_ATTENTION_ONLY -> QueryType.QT_PREFIX_MATCH;
            case WITH_MAMBA -> QueryType.QT_PREFIX_MATCH_WITH_MAMBA;
            default -> null;
        };
    }

    private void refreshCrossGroupQueryTypes() {
        for (RoleType roleType : RoleType.values()) {
            QueryType resolvedQueryType = null;
            for (ConcurrentMap<RoleType, QueryType> queryTypeByRole : queryTypeByGroupAndRole.values()) {
                QueryType candidate = queryTypeByRole.get(roleType);
                if (candidate == null) {
                    continue;
                }
                if (resolvedQueryType == null) {
                    resolvedQueryType = candidate;
                } else if (resolvedQueryType != candidate) {
                    queryTypeByRoleForCrossGroupRouting.put(roleType, DEFAULT_QUERY_TYPE);
                    log.warn("KVCM query type differs across groups for role={}; "
                            + "cross-group routing will use {}", roleType, DEFAULT_QUERY_TYPE);
                    resolvedQueryType = null;
                    break;
                }
            }
            if (resolvedQueryType != null) {
                queryTypeByRoleForCrossGroupRouting.put(roleType, resolvedQueryType);
            }
        }
    }

    private void refreshCrossGroupRollbackBlocks() {
        for (RoleType roleType : RoleType.values()) {
            Integer resolvedRollbackBlocks = null;
            for (ConcurrentMap<RoleType, Integer> rollbackBlocksByRole : rollbackBlocksByGroupAndRole.values()) {
                Integer candidate = rollbackBlocksByRole.get(roleType);
                if (candidate == null) {
                    continue;
                }
                resolvedRollbackBlocks = resolvedRollbackBlocks == null
                        ? candidate
                        : Math.max(resolvedRollbackBlocks, candidate);
            }
            if (resolvedRollbackBlocks != null) {
                rollbackBlocksByRoleForCrossGroupRouting.put(roleType, resolvedRollbackBlocks);
            }
        }
    }

    private record ResolvedWorkerMetadata(
            String namespace,
            QueryType queryType,
            Integer cacheMatchRollbackBlocks) {
    }
}
