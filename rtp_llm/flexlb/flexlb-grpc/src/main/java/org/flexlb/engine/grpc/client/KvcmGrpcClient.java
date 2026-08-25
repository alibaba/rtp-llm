package org.flexlb.engine.grpc.client;

import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.springframework.beans.factory.annotation.Autowired;
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

/** High-level KVCM cache matching client. */
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
    private final GrpcReporter grpcReporter;
    private final ScheduledExecutorService refreshExecutor;
    private final int heartbeatFailureThreshold;
    private final int queryFailureThreshold;
    private final int maxQueryRetryCount;
    private final int recoverySuccessThreshold;
    private final AtomicBoolean immediateRefreshQueued = new AtomicBoolean();
    private final AtomicReference<KvcmHealthState> healthState =
            new AtomicReference<>(KvcmHealthState.HEALTHY);
    private final AtomicInteger consecutiveHeartbeatFailures = new AtomicInteger();
    private final AtomicInteger consecutiveHeartbeatSuccesses = new AtomicInteger();
    private final AtomicInteger consecutiveQueryFailures = new AtomicInteger();
    private final AtomicLong lastHeartbeatSuccessTimeMs = new AtomicLong();
    private final AtomicLong lastHeartbeatFailureTimeMs = new AtomicLong();
    private final AtomicReference<String> lastStateChangeReason =
            new AtomicReference<>(INITIAL_HEALTH_REASON);
    private volatile Consumer<KvcmHealthSnapshot> healthSnapshotListener = ignored -> { };

    public KvcmGrpcClient(
            CacheMatchConfiguration configuration,
            KvcmMetaServiceClient metaServiceClient,
            KvcmLeaderResolver leaderResolver,
            KvcmWorkerMetadataResolver workerMetadataResolver,
            GrpcReporter grpcReporter) {
        this(configuration, metaServiceClient, leaderResolver, workerMetadataResolver,
                () -> true, grpcReporter);
    }

    @Autowired
    public KvcmGrpcClient(
            CacheMatchConfiguration configuration,
            KvcmMetaServiceClient metaServiceClient,
            KvcmLeaderResolver leaderResolver,
            KvcmWorkerMetadataResolver workerMetadataResolver,
            ApplicationWarmupState applicationWarmupState,
            GrpcReporter grpcReporter) {
        this.metaServiceClient = metaServiceClient;
        this.leaderResolver = leaderResolver;
        this.workerMetadataResolver = workerMetadataResolver;
        this.applicationWarmupState = applicationWarmupState;
        this.grpcReporter = grpcReporter;
        this.config = configuration.getKvcmConfig();
        this.enabled = configuration.isKvcmEnabled();

        if (!enabled) {
            this.heartbeatFailureThreshold = KvcmConfig.DEFAULT_HEARTBEAT_FAILURE_THRESHOLD;
            this.queryFailureThreshold = KvcmConfig.DEFAULT_QUERY_FAILURE_THRESHOLD;
            this.maxQueryRetryCount = 0;
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
                Map<String, org.flexlb.dao.cache.HostCacheMatch> result = queryOnce(
                        requestId, blockCacheKeys, namespace, queryType,
                        roleType, group, attemptIndex > 0);
                recordQuerySuccess();
                return result;
            } catch (RuntimeException failure) {
                if (attemptIndex == maxQueryRetryCount) {
                    recordQueryFailure();
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
            recordHeartbeat(leaderResolver.refresh());
        } catch (RuntimeException error) {
            log.warn("Failed to refresh KVCM leader state; keeping the last known value", error);
            recordHeartbeat(false);
        }
        try {
            workerMetadataResolver.refreshNamespacesAndQueryTypes();
        } catch (RuntimeException error) {
            log.warn("Failed to refresh KVCM metadata; keeping the last known values", error);
        }
    }

    public void setHealthSnapshotListener(Consumer<KvcmHealthSnapshot> listener) {
        this.healthSnapshotListener = listener == null ? ignored -> { } : listener;
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
        long currentTimeMs = System.currentTimeMillis();
        if (!applicationWarmupState.isWarmupFinished()) {
            if (success) {
                lastHeartbeatSuccessTimeMs.set(currentTimeMs);
            } else {
                lastHeartbeatFailureTimeMs.set(currentTimeMs);
            }
            return;
        }
        if (success) {
            recordHeartbeatSuccess(currentTimeMs);
        } else {
            recordHeartbeatFailure(currentTimeMs);
        }
        notifyHealthSnapshotListener();
    }

    private void recordHeartbeatSuccess(long currentTimeMs) {
        lastHeartbeatSuccessTimeMs.set(currentTimeMs);
        consecutiveHeartbeatFailures.set(0);
        int successes = consecutiveHeartbeatSuccesses.incrementAndGet();
        if (successes >= recoverySuccessThreshold
                && healthState.compareAndSet(KvcmHealthState.UNHEALTHY, KvcmHealthState.HEALTHY)) {
            consecutiveQueryFailures.set(0);
            recordHealthTransition("heartbeat recovery threshold reached");
        }
    }

    private void recordHeartbeatFailure(long currentTimeMs) {
        lastHeartbeatFailureTimeMs.set(currentTimeMs);
        consecutiveHeartbeatSuccesses.set(0);
        int failures = consecutiveHeartbeatFailures.incrementAndGet();
        if (failures >= heartbeatFailureThreshold
                && healthState.compareAndSet(KvcmHealthState.HEALTHY, KvcmHealthState.UNHEALTHY)) {
            recordHealthTransition("heartbeat failure threshold reached");
        }
    }

    private void recordQuerySuccess() {
        consecutiveQueryFailures.set(0);
    }

    private void recordQueryFailure() {
        if (!applicationWarmupState.isWarmupFinished()) {
            return;
        }
        int failures = consecutiveQueryFailures.incrementAndGet();
        if (failures >= queryFailureThreshold
                && healthState.compareAndSet(KvcmHealthState.HEALTHY, KvcmHealthState.UNHEALTHY)) {
            consecutiveHeartbeatSuccesses.set(0);
            recordHealthTransition("cache query failure threshold reached");
            notifyHealthSnapshotListener();
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
        } catch (RuntimeException error) {
            log.error("KVCM health snapshot listener failed, state={}", snapshot.state(), error);
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
