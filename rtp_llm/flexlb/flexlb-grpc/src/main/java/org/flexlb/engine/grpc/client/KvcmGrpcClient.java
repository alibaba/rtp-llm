package org.flexlb.engine.grpc.client;

import io.grpc.StatusRuntimeException;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.listener.ApplicationWarmupState;
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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

/**
 * High-level KVCM cache matching client.
 */
@Slf4j
@Component
public class KvcmGrpcClient {

    private static final String INITIAL_HEALTH_REASON = "initial";

    private final boolean enabled;
    private final KvcmConfig config;
    private final KvcmMetaServiceClient metaServiceClient;
    private final KvcmLeaderResolver leaderResolver;
    private final KvcmWorkerMetadataResolver workerMetadataResolver;
    private final ApplicationWarmupState applicationWarmupState;
    private final ScheduledExecutorService refreshExecutor;
    private final int heartbeatFailureThreshold;
    private final int queryFailureThreshold;
    private final int maxQueryRetryCount;
    private final int recoverySuccessThreshold;
    private final AtomicBoolean immediateRefreshQueued = new AtomicBoolean();
    private final AtomicReference<KvcmHealthState> healthState = new AtomicReference<>(KvcmHealthState.HEALTHY);
    private final AtomicInteger consecutiveHeartbeatFailures = new AtomicInteger();
    private final AtomicInteger consecutiveHeartbeatSuccesses = new AtomicInteger();
    private final AtomicInteger consecutiveQueryFailures = new AtomicInteger();
    private final AtomicLong lastHeartbeatSuccessTimeMs = new AtomicLong();
    private final AtomicLong lastHeartbeatFailureTimeMs = new AtomicLong();
    private final AtomicReference<String> lastStateChangeReason = new AtomicReference<>(INITIAL_HEALTH_REASON);
    @Setter
    private volatile Consumer<KvcmHealthSnapshot> healthSnapshotListener = ignored -> { };

    public KvcmGrpcClient(CacheMatchConfiguration configuration,
                          KvcmMetaServiceClient metaServiceClient,
                          KvcmLeaderResolver leaderResolver,
                          KvcmWorkerMetadataResolver workerMetadataResolver,
                          ApplicationWarmupState applicationWarmupState) {
        this.metaServiceClient = metaServiceClient;
        this.leaderResolver = leaderResolver;
        this.workerMetadataResolver = workerMetadataResolver;
        this.applicationWarmupState = applicationWarmupState;
        this.config = configuration.getKvcmConfig();
        this.enabled = configuration.isKvcmEnabled();

        if (!enabled) {
            this.heartbeatFailureThreshold = KvcmConfig.DEFAULT_HEARTBEAT_FAILURE_THRESHOLD;
            this.queryFailureThreshold = KvcmConfig.DEFAULT_QUERY_FAILURE_THRESHOLD;
            this.maxQueryRetryCount = KvcmConfig.DEFAULT_MAX_QUERY_RETRY_COUNT;
            this.recoverySuccessThreshold = KvcmConfig.DEFAULT_RECOVERY_SUCCESS_THRESHOLD;
            this.refreshExecutor = null;
            return;
        }

        this.heartbeatFailureThreshold = config.getHeartbeatFailureThreshold();
        this.queryFailureThreshold = config.getQueryFailureThreshold();
        this.maxQueryRetryCount = Math.max(0, config.getMaxQueryRetryCount());
        this.recoverySuccessThreshold = config.getRecoverySuccessThreshold();
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
        log.info("Started KVCM client, address={}, bootstrapPort={}, leaderRefreshIntervalMs={}, "
                        + "heartbeatFailureThreshold={}, queryFailureThreshold={}, maxQueryRetryCount={}, "
                        + "recoverySuccessThreshold={}, namespaceSource={}",
                config.getAddress(),
                config.getPort(),
                config.getLeaderRefreshIntervalMs(),
                heartbeatFailureThreshold,
                queryFailureThreshold,
                maxQueryRetryCount,
                recoverySuccessThreshold,
                workerMetadataResolver.usesConfiguredNamespace() ? "configuration" : "worker-status");
    }

    public Map<String, Integer> findMatchingEngines(String requestId, List<Long> blockCacheKeys, long blockSize,
                                                    RoleType roleType, String group) {
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
        if (blockSize <= 0) {
            log.warn("Skipping KVCM cache query because blockSize is unavailable, "
                            + "requestId={}, role={}, group={}",
                    requestId, roleType, group);
            return Collections.emptyMap();
        }

        String namespace = workerMetadataResolver.resolveNamespace(roleType, group, blockSize);
        if (StringUtils.isBlank(namespace)) {
            requestImmediateRefresh();
            log.warn("Skipping KVCM cache query because namespace is unavailable, "
                            + "requestId={}, role={}, group={}",
                    requestId, roleType, group);
            return Collections.emptyMap();
        }
        QueryType queryType = workerMetadataResolver.resolveQueryType(roleType, group);
        if (queryType == null) {
            requestImmediateRefresh();
            log.warn("Skipping KVCM cache query because query type is unavailable, "
                    + "requestId={}, role={}, group={}", requestId, roleType, group);
            return Collections.emptyMap();
        }
        int rollbackBlocks = workerMetadataResolver.resolveCacheMatchRollbackBlocks(roleType, group);
        return queryWithRetry(requestId, blockCacheKeys, namespace, queryType, rollbackBlocks, roleType, group);
    }

    private Map<String, Integer> queryWithRetry(String requestId, List<Long> blockCacheKeys, String namespace,
                                                QueryType queryType, int rollbackBlocks,
                                                RoleType roleType, String group) {
        for (int attemptIndex = 0; attemptIndex <= maxQueryRetryCount; attemptIndex++) {
            try {
                Map<String, Integer> matches = queryOnce(
                        requestId, blockCacheKeys, namespace, queryType, rollbackBlocks, roleType, group);
                recordQuerySuccess();
                return matches;
            } catch (RuntimeException failure) {
                if (attemptIndex == maxQueryRetryCount) {
                    recordQueryFailure();
                    throw failure;
                }
                log.debug("KVCM cache query failed; retrying, requestId={}, attempt={}, maxRetryCount={}",
                        requestId, attemptIndex + 1, maxQueryRetryCount, failure);
            }
        }
        throw new IllegalStateException("KVCM query retry loop completed without a result");
    }

    private Map<String, Integer> queryOnce(String requestId, List<Long> blockCacheKeys, String namespace,
                                           QueryType queryType, int rollbackBlocks,
                                           RoleType roleType, String group) {
        GrpcTarget currentLeader = leaderResolver.resolve();
        if (currentLeader == null) {
            throw new KvcmQueryException("KVCM leader is unavailable");
        }

        String traceId = IdUtils.fastUuid();
        GetHostCacheStateRequest request = GetHostCacheStateRequest.newBuilder()
                .setTraceId(traceId)
                // KVCM exposes the cache namespace as instance_id in its protocol.
                .setInstanceId(namespace)
                .setQueryType(queryType)
                .addAllBlockCacheKeys(blockCacheKeys)
                .setUseEaglePop(rollbackBlocks > 0)
                .build();

        try {
            if (log.isDebugEnabled()) {
                log.debug("KVCM GetHostCacheState request: requestId={}, traceId={}, namespace={}, "
                                + "leader={}, role={}, group={}, queryType={}, useEaglePop={}, "
                                + "blockCount={}, blockCacheKeys={}",
                        requestId, traceId, namespace, currentLeader, roleType, group,
                        queryType, request.getUseEaglePop(), blockCacheKeys.size(), blockCacheKeys);
            }
            GetHostCacheStateResponse response = metaServiceClient.getHostCacheState(
                    currentLeader, request, config.getRequestTimeoutMs());
            ErrorCode code = response.getHeader().getStatus().getCode();
            if (code != ErrorCode.OK) {
                throw new KvcmQueryException(
                        "KVCM GetHostCacheState failed, code=" + code + ", message="
                                + response.getHeader().getStatus().getMessage());
            }
            Map<String, Integer> matches = toPrefixMatchBlocksByHost(response.getHostsList());
            if (log.isDebugEnabled()) {
                log.debug("KVCM GetHostCacheState response: requestId={}, traceId={}, matches={}",
                        requestId, traceId, matches);
            }
            return matches;
        } catch (StatusRuntimeException e) {
            throw new KvcmQueryException("KVCM GetHostCacheState gRPC request failed", e);
        }
    }

    void refreshKvcmServiceStateSafely() {
        try {
            recordHeartbeat(leaderResolver.refresh());
        } catch (Exception e) {
            log.warn("Failed to refresh KVCM leader state; keeping the last known value", e);
            recordHeartbeat(false);
        }
        try {
            workerMetadataResolver.refreshNamespacesAndQueryTypes();
        } catch (Exception e) {
            log.warn("Failed to refresh KVCM namespaces and query types; keeping the last known values", e);
        }
    }

    public KvcmHealthSnapshot healthSnapshot() {
        return new KvcmHealthSnapshot(
                healthState.get(),
                consecutiveHeartbeatFailures.get(),
                consecutiveHeartbeatSuccesses.get(),
                consecutiveQueryFailures.get(),
                lastHeartbeatSuccessTimeMs.get(),
                lastHeartbeatFailureTimeMs.get(),
                lastStateChangeReason.get());
    }

    private void recordHeartbeat(boolean success) {
        if (!applicationWarmupState.isWarmupFinished()) {
            long currentTimeMs = System.currentTimeMillis();
            if (success) {
                lastHeartbeatSuccessTimeMs.set(currentTimeMs);
            } else {
                lastHeartbeatFailureTimeMs.set(currentTimeMs);
            }
            log.debug("Ignoring KVCM heartbeat result during application warm-up, success={}", success);
            return;
        }
        if (success) {
            recordHeartbeatSuccess();
        } else {
            recordHeartbeatFailure();
        }
        notifyHealthSnapshotListener();
    }

    private void recordHeartbeatSuccess() {
        lastHeartbeatSuccessTimeMs.set(System.currentTimeMillis());
        consecutiveHeartbeatFailures.set(0);
        int successes = consecutiveHeartbeatSuccesses.incrementAndGet();
        if (successes >= recoverySuccessThreshold) {
            boolean healthRecovered = healthState.compareAndSet(KvcmHealthState.UNHEALTHY, KvcmHealthState.HEALTHY);
            if (healthRecovered) {
                consecutiveQueryFailures.set(0);
                recordHealthTransition("heartbeat recovery threshold reached");
            }
        }
    }

    private void recordHeartbeatFailure() {
        lastHeartbeatFailureTimeMs.set(System.currentTimeMillis());
        consecutiveHeartbeatSuccesses.set(0);
        int failures = consecutiveHeartbeatFailures.incrementAndGet();
        if (failures >= heartbeatFailureThreshold) {
            boolean becameUnhealthy = healthState.compareAndSet(KvcmHealthState.HEALTHY, KvcmHealthState.UNHEALTHY);
            if (becameUnhealthy) {
                recordHealthTransition("heartbeat failure threshold reached");
            }
        }
    }

    private void recordQuerySuccess() {
        consecutiveQueryFailures.set(0);
    }

    private void recordQueryFailure() {
        if (!applicationWarmupState.isWarmupFinished()) {
            log.debug("Ignoring KVCM query failure during application warm-up");
            return;
        }
        int failures = consecutiveQueryFailures.incrementAndGet();
        if (failures >= queryFailureThreshold) {
            boolean becameUnhealthy = healthState.compareAndSet(KvcmHealthState.HEALTHY, KvcmHealthState.UNHEALTHY);
            if (becameUnhealthy) {
                consecutiveHeartbeatSuccesses.set(0);
                recordHealthTransition("cache query failure threshold reached");
                notifyHealthSnapshotListener();
            }
        }
    }

    private void recordHealthTransition(String reason) {
        lastStateChangeReason.set(reason);
        KvcmHealthSnapshot snapshot = healthSnapshot();
        if (snapshot.isHealthy()) {
            log.info("KVCM health recovered, reason={}, consecutiveHeartbeatSuccesses={}",
                    reason, snapshot.consecutiveHeartbeatSuccesses());
        } else {
            log.warn("KVCM marked unhealthy, reason={}, consecutiveHeartbeatFailures={}, "
                            + "consecutiveQueryFailures={}",
                    reason,
                    snapshot.consecutiveHeartbeatFailures(),
                    snapshot.consecutiveQueryFailures());
        }
    }

    private void notifyHealthSnapshotListener() {
        KvcmHealthSnapshot snapshot = healthSnapshot();
        try {
            healthSnapshotListener.accept(snapshot);
        } catch (RuntimeException e) {
            log.error("KVCM health snapshot listener failed, state={}", snapshot.state(), e);
        }
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

    private void requestImmediateRefresh() {
        if (refreshExecutor == null || refreshExecutor.isShutdown()
                || !immediateRefreshQueued.compareAndSet(false, true)) {
            return;
        }
        try {
            refreshExecutor.execute(() -> {
                try {
                    workerMetadataResolver.refreshNamespacesAndQueryTypes();
                } catch (Exception e) {
                    log.warn("Failed to refresh KVCM namespaces and query types; keeping the last known values", e);
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
