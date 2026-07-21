package org.flexlb.engine.grpc.client;

import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * High-level KVCM cache matching client.
 */
@Slf4j
@Component
public class KvcmGrpcClient {

    private final boolean enabled;
    private final KvcmConfig config;
    private final KvcmMetaServiceClient metaServiceClient;
    private final KvcmLeaderResolver leaderResolver;
    private final KvcmNamespaceResolver namespaceResolver;
    private final KvcmQueryTypeResolver queryTypeResolver;
    private final ScheduledExecutorService refreshExecutor;
    private final AtomicBoolean immediateRefreshQueued = new AtomicBoolean();

    public KvcmGrpcClient(
            ModelMetaConfig modelMetaConfig,
            KvcmMetaServiceClient metaServiceClient,
            KvcmLeaderResolver leaderResolver,
            KvcmNamespaceResolver namespaceResolver,
            KvcmQueryTypeResolver queryTypeResolver) {
        this.metaServiceClient = metaServiceClient;
        this.leaderResolver = leaderResolver;
        this.namespaceResolver = namespaceResolver;
        this.queryTypeResolver = queryTypeResolver;
        ServiceRoute serviceRoute = modelMetaConfig.getServiceRoutes().stream().findFirst().orElse(null);
        this.config = serviceRoute != null ? serviceRoute.getKvcm() : null;
        this.enabled = config != null && config.isEnabled();

        if (!enabled) {
            this.refreshExecutor = null;
            return;
        }

        this.refreshExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "kvcm-service-state-refresher");
            thread.setDaemon(true);
            return thread;
        });
        this.refreshExecutor.scheduleWithFixedDelay(
                this::refreshKvcmServiceStateSafely,
                0,
                config.getLeaderRefreshIntervalMs(),
                TimeUnit.MILLISECONDS);
        log.info("Started KVCM client, address={}, bootstrapPort={}, leaderRefreshIntervalMs={}, namespaceSource={}",
                config.getAddress(), config.getPort(), config.getLeaderRefreshIntervalMs(),
                namespaceResolver.usesConfiguredNamespace() ? "configuration" : "worker-discovery");
    }

    public Map<String, Integer> findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            RoleType roleType,
            String group) {
        if (!enabled) {
            log.warn("Skipping KVCM cache query because the KVCM client is disabled, "
                            + "requestId={}, role={}, group={}",
                    requestId, roleType, group);
            return Collections.emptyMap();
        }
        if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
            log.debug("Skipping KVCM cache query because blockCacheKeys is empty, requestId={}", requestId);
            return Collections.emptyMap();
        }

        String namespace = namespaceResolver.resolve(roleType, group);
        if (StringUtils.isBlank(namespace)) {
            log.warn("Skipping KVCM cache query because namespace is unavailable, "
                            + "requestId={}, role={}, group={}",
                    requestId, roleType, group);
            requestImmediateRefresh();
            return Collections.emptyMap();
        }
        GrpcTarget currentLeader = leaderResolver.resolve();
        if (currentLeader == null) {
            log.warn("Skipping KVCM cache query because leader is unavailable, "
                            + "requestId={}, role={}, group={}",
                    requestId, roleType, group);
            requestImmediateRefresh();
            return Collections.emptyMap();
        }

        String traceId = IdUtils.fastUuid();
        QueryType queryType = queryTypeResolver.resolve(roleType, group);
        GetHostCacheStateRequest request = GetHostCacheStateRequest.newBuilder()
                .setTraceId(traceId)
                // KVCM exposes the cache namespace as instance_id in its protocol.
                .setInstanceId(namespace)
                .setQueryType(queryType)
                .addAllBlockCacheKeys(blockCacheKeys)
                .build();

        try {
            if (log.isDebugEnabled()) {
                log.debug("KVCM GetHostCacheState request: requestId={}, traceId={}, namespace={}, "
                                + "leader={}, role={}, group={}, queryType={}, blockCount={}, blockCacheKeys={}",
                        requestId, traceId, namespace, currentLeader, roleType, group, queryType,
                        blockCacheKeys.size(), blockCacheKeys);
            }
            GetHostCacheStateResponse response = metaServiceClient.getHostCacheState(
                    currentLeader, request, config.getRequestTimeoutMs());
            ErrorCode code = response.getHeader().getStatus().getCode();
            if (code != ErrorCode.OK) {
                if (code == ErrorCode.SERVER_NOT_LEADER || code == ErrorCode.INSTANCE_NOT_EXIST) {
                    requestImmediateRefresh();
                }
                throw new IllegalStateException(
                        "KVCM GetHostCacheState failed, code=" + code
                                + ", message=" + response.getHeader().getStatus().getMessage());
            }
            Map<String, Integer> matches = toPrefixMatchBlocksByHost(response.getHostsList());
            if (log.isDebugEnabled()) {
                log.debug("KVCM GetHostCacheState response: requestId={}, traceId={}, matches={}",
                        requestId, traceId, matches);
            }
            return matches;
        } catch (StatusRuntimeException e) {
            requestImmediateRefresh();
            throw e;
        }
    }

    private void refreshKvcmServiceStateSafely() {
        refreshSafely("query type state", queryTypeResolver::refresh);
        refreshSafely("leader state", leaderResolver::refresh);
        refreshSafely("namespace state", namespaceResolver::refresh);
    }

    private Map<String, Integer> toPrefixMatchBlocksByHost(List<HostCacheMatch> matches) {
        Map<String, Integer> result = new HashMap<>();
        for (HostCacheMatch match : matches) {
            if (StringUtils.isBlank(match.getHostIpPort())) {
                continue;
            }
            int prefixMatchBlocks = Math.toIntExact(match.getPrefixMatchBlocks());
            result.merge(match.getHostIpPort(), prefixMatchBlocks, Math::max);
        }
        return result;
    }

    private void refreshSafely(String stateName, Runnable refreshAction) {
        try {
            refreshAction.run();
        } catch (Exception e) {
            log.warn("Failed to refresh KVCM {}; keeping the last known value", stateName, e);
        }
    }

    private void requestImmediateRefresh() {
        if (refreshExecutor == null || refreshExecutor.isShutdown()
                || !immediateRefreshQueued.compareAndSet(false, true)) {
            return;
        }
        try {
            refreshExecutor.execute(() -> {
                try {
                    refreshKvcmServiceStateSafely();
                } finally {
                    immediateRefreshQueued.set(false);
                }
            });
        } catch (RejectedExecutionException e) {
            immediateRefreshQueued.set(false);
        }
    }

    @PreDestroy
    public void shutdown() {
        if (refreshExecutor != null) {
            refreshExecutor.shutdown();
        }
        metaServiceClient.shutdown();
    }

}
