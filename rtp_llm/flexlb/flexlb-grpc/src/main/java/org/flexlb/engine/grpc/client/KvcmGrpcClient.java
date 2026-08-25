package org.flexlb.engine.grpc.client;

import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
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

/** High-level KVCM cache matching client. */
@Slf4j
@Component
public class KvcmGrpcClient {

    private final boolean enabled;
    private final KvcmConfig config;
    private final KvcmMetaServiceClient metaServiceClient;
    private final KvcmLeaderResolver leaderResolver;
    private final KvcmWorkerMetadataResolver workerMetadataResolver;
    private final GrpcReporter grpcReporter;
    private final ScheduledExecutorService refreshExecutor;
    private final int maxQueryRetryCount;
    private final AtomicBoolean immediateRefreshQueued = new AtomicBoolean();

    public KvcmGrpcClient(
            CacheMatchConfiguration configuration,
            KvcmMetaServiceClient metaServiceClient,
            KvcmLeaderResolver leaderResolver,
            KvcmWorkerMetadataResolver workerMetadataResolver,
            GrpcReporter grpcReporter) {
        this.metaServiceClient = metaServiceClient;
        this.leaderResolver = leaderResolver;
        this.workerMetadataResolver = workerMetadataResolver;
        this.grpcReporter = grpcReporter;
        this.config = configuration.getKvcmConfig();
        this.enabled = configuration.isKvcmEnabled();

        if (!enabled) {
            this.maxQueryRetryCount = 0;
            this.refreshExecutor = null;
            return;
        }

        this.maxQueryRetryCount = Math.max(0, config.getMaxQueryRetryCount());
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
        log.info("Started KVCM client, address={}, bootstrapPort={}, "
                        + "leaderRefreshIntervalMs={}, maxQueryRetryCount={}, namespaceSource={}",
                config.getAddress(), config.getPort(), config.getLeaderRefreshIntervalMs(),
                maxQueryRetryCount,
                workerMetadataResolver.usesConfiguredNamespace()
                        ? "configuration"
                        : "worker-status");
    }

    public Map<String, org.flexlb.dao.cache.HostCacheMatch> findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            long blockSize,
            RoleType roleType,
            String group) {
        if (!enabled) {
            return Collections.emptyMap();
        }
        if (blockCacheKeys == null || blockCacheKeys.isEmpty() || blockSize <= 0) {
            return Collections.emptyMap();
        }

        String namespace = workerMetadataResolver.resolveNamespace(roleType, group, blockSize);
        QueryType queryType = workerMetadataResolver.resolveQueryType(roleType, group);
        if (StringUtils.isBlank(namespace) || queryType == null) {
            requestImmediateRefresh();
            return Collections.emptyMap();
        }
        return queryWithRetry(
                requestId, blockCacheKeys, namespace, queryType, roleType, group);
    }

    private Map<String, org.flexlb.dao.cache.HostCacheMatch> queryWithRetry(
            String requestId,
            List<Long> blockCacheKeys,
            String namespace,
            QueryType queryType,
            RoleType roleType,
            String group) {
        for (int attemptIndex = 0; attemptIndex <= maxQueryRetryCount; attemptIndex++) {
            try {
                return queryOnce(
                        requestId, blockCacheKeys, namespace, queryType,
                        roleType, group, attemptIndex > 0);
            } catch (RuntimeException failure) {
                if (attemptIndex == maxQueryRetryCount) {
                    throw failure;
                }
                log.debug("KVCM cache query failed; retrying, requestId={}, "
                                + "attempt={}, maxRetryCount={}",
                        requestId, attemptIndex + 1, maxQueryRetryCount, failure);
            }
        }
        throw new IllegalStateException("KVCM query retry loop completed without a result");
    }

    private Map<String, org.flexlb.dao.cache.HostCacheMatch> queryOnce(
            String requestId,
            List<Long> blockCacheKeys,
            String namespace,
            QueryType queryType,
            RoleType roleType,
            String group,
            boolean retry) {
        GrpcTarget currentLeader = leaderResolver.resolve();
        if (currentLeader == null) {
            requestImmediateRefresh();
            throw new KvcmQueryException("KVCM leader is unavailable");
        }

        GetHostCacheStateRequest request = GetHostCacheStateRequest.newBuilder()
                .setTraceId(requestId)
                .setInstanceId(namespace)
                .setQueryType(queryType)
                .addAllBlockCacheKeys(blockCacheKeys)
                .setP2PHostCount(Math.max(0, config.getP2pHostCount()))
                .build();

        try {
            long startTimeUs = System.nanoTime() / 1_000;
            GetHostCacheStateResponse response = metaServiceClient.getHostCacheState(
                    currentLeader, request, config.getRequestTimeoutMs());
            grpcReporter.reportCallMetrics(
                    "KVCM_GET_HOST_CACHE_STATE",
                    System.nanoTime() / 1_000 - startTimeUs,
                    response.getSerializedSize(),
                    retry);
            ErrorCode code = response.getHeader().getStatus().getCode();
            if (code != ErrorCode.OK) {
                requestImmediateRefresh();
                throw new KvcmQueryException(
                        "KVCM GetHostCacheState failed, code=" + code
                                + ", message="
                                + response.getHeader().getStatus().getMessage());
            }
            return toMatchesByHost(response.getHostsList());
        } catch (StatusRuntimeException error) {
            requestImmediateRefresh();
            throw new KvcmQueryException(
                    "KVCM GetHostCacheState gRPC request failed", error);
        }
    }

    void refreshKvcmServiceStateSafely() {
        try {
            leaderResolver.refresh();
        } catch (RuntimeException error) {
            log.warn("Failed to refresh KVCM leader state; keeping the last known value", error);
        }
        try {
            workerMetadataResolver.refreshNamespacesAndQueryTypes();
        } catch (RuntimeException error) {
            log.warn("Failed to refresh KVCM metadata; keeping the last known values", error);
        }
    }

    private Map<String, org.flexlb.dao.cache.HostCacheMatch> toMatchesByHost(
            List<HostCacheMatch> matches) {
        Map<String, org.flexlb.dao.cache.HostCacheMatch> result = new HashMap<>();
        for (HostCacheMatch match : matches) {
            if (StringUtils.isBlank(match.getHostIpPort())) {
                continue;
            }
            result.put(
                    match.getHostIpPort(),
                    new org.flexlb.dao.cache.HostCacheMatch(
                            match.getLocal(),
                            match.getP2P1Fetch(),
                            match.getP2P1TotalMatch()));
        }
        return result;
    }

    private void requestImmediateRefresh() {
        if (refreshExecutor == null
                || refreshExecutor.isShutdown()
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
        } catch (RejectedExecutionException error) {
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
