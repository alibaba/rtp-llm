package org.flexlb.engine.grpc.client;

import io.grpc.Deadline;
import io.grpc.StatusRuntimeException;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.dao.kvcm.KvcmHealthSnapshot;
import org.flexlb.dao.kvcm.KvcmHealthState;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.monitor.KvcmMetricsReporter;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.listener.ApplicationWarmupState;
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
    private static final long MIN_IMMEDIATE_REFRESH_INTERVAL_NANOS = TimeUnit.SECONDS.toNanos(1);

    private final boolean enabled;
    private final CacheMatchConfiguration configuration;
    private final KvcmConfig topologyConfig;
    private final KvcmMetaServiceClient metaServiceClient;
    private final KvcmLeaderResolver leaderResolver;
    private final KvcmWorkerMetadataResolver workerMetadataResolver;
    private final ApplicationWarmupState applicationWarmupState;
    private final GrpcReporter grpcReporter;
    private final KvcmMetricsReporter metricsReporter;
    private final ScheduledExecutorService refreshExecutor;
    private final AtomicBoolean immediateRefreshQueued = new AtomicBoolean();
    private final AtomicLong lastImmediateRefreshScheduledNs = new AtomicLong(Long.MIN_VALUE);
    private final AtomicReference<KvcmHealthState> healthState =
            new AtomicReference<>(KvcmHealthState.HEALTHY);
    private final AtomicInteger consecutiveHeartbeatFailures = new AtomicInteger();
    private final AtomicInteger consecutiveHeartbeatSuccesses = new AtomicInteger();
    private final AtomicInteger consecutiveQueryFailures = new AtomicInteger();
    private final AtomicInteger consecutiveQuerySuccesses = new AtomicInteger();
    private final AtomicLong lastHeartbeatSuccessTimeMs = new AtomicLong();
    private final AtomicLong lastHeartbeatFailureTimeMs = new AtomicLong();
    private final AtomicReference<String> lastStateChangeReason =
            new AtomicReference<>(INITIAL_HEALTH_REASON);
    private volatile Consumer<KvcmHealthSnapshot> healthSnapshotListener = ignored -> { };

    public KvcmGrpcClient(CacheMatchConfiguration configuration,
                          KvcmMetaServiceClient metaServiceClient,
                          KvcmLeaderResolver leaderResolver,
                          KvcmWorkerMetadataResolver workerMetadataResolver,
                          ApplicationWarmupState applicationWarmupState,
                          GrpcReporter grpcReporter,
                          KvcmMetricsReporter metricsReporter) {
        this.configuration = configuration;
        this.metaServiceClient = metaServiceClient;
        this.leaderResolver = leaderResolver;
        this.workerMetadataResolver = workerMetadataResolver;
        this.applicationWarmupState = applicationWarmupState;
        this.grpcReporter = grpcReporter;
        this.metricsReporter = metricsReporter;
        this.topologyConfig = configuration.getKvcmConfig();
        this.enabled = configuration.isKvcmEnabled();

        if (!enabled) {
            this.refreshExecutor = null;
            return;
        }

        KvcmCacheMatchingConfig startupConfig = configuration.getKvcmRuntimeConfig();
        this.refreshExecutor = Executors.newSingleThreadScheduledExecutor(runnable -> {
            Thread thread = new Thread(runnable, "kvcm-service-state-refresher");
            thread.setDaemon(true);
            return thread;
        });
        this.refreshExecutor.scheduleWithFixedDelay(
                this::refreshKvcmServiceStateSafely,
                0,
                startupConfig.getLeaderRefreshIntervalMs(),
                TimeUnit.MILLISECONDS);
        log.info("Started KVCM client, address={}, bootstrapPort={}, "
                        + "leaderRefreshIntervalMs={}, maxQueryRetryCount={}, namespaceSource={}",
                topologyConfig.getAddress(), topologyConfig.getPort(),
                startupConfig.getLeaderRefreshIntervalMs(),
                startupConfig.getMaxQueryRetryCount(),
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
            recordQueryFailure();
            metricsReporter.reportQueryFailure();
            throw new KvcmQueryException("KVCM worker metadata unavailable for role=" + roleType
                    + ", group=" + group);
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
        // One snapshot per query keeps the wire parameters of every attempt consistent.
        KvcmCacheMatchingConfig config = configuration.getKvcmRuntimeConfig();
        int maxQueryRetryCount = Math.max(0, config.getMaxQueryRetryCount());
        Deadline queryDeadline = Deadline.after(config.getRequestTimeoutMs(), TimeUnit.MILLISECONDS);
        for (int attemptIndex = 0; ; attemptIndex++) {
            try {
                if (queryDeadline.isExpired()) {
                    throw new KvcmQueryException("KVCM cache query timed out");
                }
                Map<String, org.flexlb.dao.cache.HostCacheMatch> result = queryOnce(
                        config, queryDeadline, requestId, blockCacheKeys, namespace, queryType,
                        roleType, group, attemptIndex > 0);
                recordQuerySuccess();
                return result;
            } catch (RuntimeException failure) {
                if (attemptIndex == maxQueryRetryCount || queryDeadline.isExpired()) {
                    recordQueryFailure();
                    metricsReporter.reportQueryFailure();
                    throw failure;
                }
                metricsReporter.reportQueryRetry(attemptIndex + 1);
                log.debug("KVCM cache query failed; retrying, requestId={}, "
                                + "attempt={}, maxRetryCount={}",
                        requestId, attemptIndex + 1, maxQueryRetryCount, failure);
            }
        }
    }

    private Map<String, org.flexlb.dao.cache.HostCacheMatch> queryOnce(
            KvcmCacheMatchingConfig config,
            Deadline queryDeadline,
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
                .addAllMedium(config.getMedium())
                .setGlobalKvsHostCount(Math.max(0, config.getGlobalKvsHostCount()))
                .setEnableP2P(config.isEnableP2p())
                .build();

        long startTimeNanos = System.nanoTime();
        int responseBytes = 0;
        GetHostCacheStateResponse response;
        try {
            response = metaServiceClient.getHostCacheState(
                    currentLeader, request, queryDeadline);
            responseBytes = response.getSerializedSize();
        } catch (StatusRuntimeException error) {
            requestImmediateRefresh();
            throw new KvcmQueryException(
                    "KVCM GetHostCacheState gRPC request failed", error);
        } finally {
            try {
                grpcReporter.reportCallMetrics(
                        "KVCM_GET_HOST_CACHE_STATE",
                        TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startTimeNanos),
                        responseBytes,
                        retry);
            } catch (RuntimeException error) {
                log.warn("Failed to report KVCM gRPC call metrics", error);
            }
        }
        ErrorCode code = response.getHeader().getStatus().getCode();
        if (code != ErrorCode.OK) {
            requestImmediateRefresh();
            throw new KvcmQueryException(
                    "KVCM GetHostCacheState failed, code=" + code
                            + ", message="
                            + response.getHeader().getStatus().getMessage());
        }
        return toMatchesByHost(response.getHostsList());
    }

    void refreshKvcmServiceStateSafely() {
        try {
            recordHeartbeat(leaderResolver.refresh());
        } catch (RuntimeException error) {
            log.warn("Failed to refresh KVCM leader state; keeping the last known value", error);
            recordHeartbeat(false);
        }
        if (refreshExecutor != null && refreshExecutor.isShutdown()) {
            return;
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
        if (successes >= configuration.getKvcmRuntimeConfig().getRecoverySuccessThreshold()
                && healthState.compareAndSet(KvcmHealthState.UNHEALTHY, KvcmHealthState.HEALTHY)) {
            consecutiveQueryFailures.set(0);
            consecutiveQuerySuccesses.set(0);
            recordHealthTransition("heartbeat recovery threshold reached");
        }
    }

    private void recordHeartbeatFailure(long currentTimeMs) {
        lastHeartbeatFailureTimeMs.set(currentTimeMs);
        consecutiveHeartbeatSuccesses.set(0);
        consecutiveQuerySuccesses.set(0);
        int failures = consecutiveHeartbeatFailures.incrementAndGet();
        if (failures >= configuration.getKvcmRuntimeConfig().getHeartbeatFailureThreshold()
                && healthState.compareAndSet(KvcmHealthState.HEALTHY, KvcmHealthState.UNHEALTHY)) {
            recordHealthTransition("heartbeat failure threshold reached");
        }
    }

    private void recordQuerySuccess() {
        consecutiveQueryFailures.set(0);
        if (!applicationWarmupState.isWarmupFinished()
                || healthState.get() == KvcmHealthState.HEALTHY) {
            consecutiveQuerySuccesses.set(0);
            return;
        }
        int successes = consecutiveQuerySuccesses.incrementAndGet();
        if (successes >= configuration.getKvcmRuntimeConfig().getRecoverySuccessThreshold()
                && healthState.compareAndSet(KvcmHealthState.UNHEALTHY, KvcmHealthState.HEALTHY)) {
            consecutiveHeartbeatFailures.set(0);
            consecutiveHeartbeatSuccesses.set(0);
            recordHealthTransition("cache query recovery threshold reached");
            notifyHealthSnapshotListener();
        }
    }

    private void recordQueryFailure() {
        consecutiveQuerySuccesses.set(0);
        if (!applicationWarmupState.isWarmupFinished()) {
            return;
        }
        consecutiveHeartbeatSuccesses.set(0);
        int failures = consecutiveQueryFailures.incrementAndGet();
        if (failures >= configuration.getKvcmRuntimeConfig().getQueryFailureThreshold()
                && healthState.compareAndSet(KvcmHealthState.HEALTHY, KvcmHealthState.UNHEALTHY)) {
            recordHealthTransition("cache query failure threshold reached");
            notifyHealthSnapshotListener();
        }
    }

    private void recordHealthTransition(String reason) {
        lastStateChangeReason.set(reason);
        KvcmHealthSnapshot snapshot = healthSnapshot();
        if (snapshot.isHealthy()) {
            log.info("KVCM health recovered, reason={}, consecutiveHeartbeatSuccesses={}, consecutiveQuerySuccesses={}",
                    reason, snapshot.consecutiveHeartbeatSuccesses(), consecutiveQuerySuccesses.get());
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
            org.flexlb.dao.cache.HostCacheMatch previous = result.put(
                    match.getHostIpPort(),
                    new org.flexlb.dao.cache.HostCacheMatch(
                            match.getLocal(),
                            match.getGlobal()));
            if (previous != null) {
                log.warn("KVCM returned duplicate cache matches for host {}; keeping the last record",
                        match.getHostIpPort());
            }
        }
        return result;
    }

    private void requestImmediateRefresh() {
        if (refreshExecutor == null
                || refreshExecutor.isShutdown()
                || !immediateRefreshQueued.compareAndSet(false, true)) {
            return;
        }
        long nowNs = System.nanoTime();
        long lastRefreshNs = lastImmediateRefreshScheduledNs.get();
        if (lastRefreshNs != Long.MIN_VALUE
                && nowNs - lastRefreshNs < MIN_IMMEDIATE_REFRESH_INTERVAL_NANOS) {
            immediateRefreshQueued.set(false);
            return;
        }
        lastImmediateRefreshScheduledNs.set(nowNs);
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
            refreshExecutor.shutdownNow();
            try {
                if (!refreshExecutor.awaitTermination(1, TimeUnit.SECONDS)) {
                    log.warn("KVCM service-state refresher did not stop within 1 second");
                }
            } catch (InterruptedException error) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
