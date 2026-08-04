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
import org.flexlb.engine.grpc.monitor.KvcmMetricsReporter;
import org.flexlb.exception.KvcmQueryException;
import org.flexlb.kvcm.grpc.ErrorCode;
import org.flexlb.kvcm.grpc.GetHostCacheStateRequest;
import org.flexlb.kvcm.grpc.GetHostCacheStateResponse;
import org.flexlb.kvcm.grpc.HostCacheMatch;
import org.flexlb.kvcm.grpc.QueryType;
import org.flexlb.listener.ApplicationWarmupState;
import org.flexlb.util.IdUtils;
import org.springframework.stereotype.Component;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

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
 * High-level reactive client for KVCM cache matching.
 *
 * <p>When KVCM is enabled, the client refreshes leader and worker metadata on a dedicated
 * background executor. The request path only reads the resulting cached namespace and query type;
 * it never performs worker discovery or metadata traversal. The current cached leader is read for
 * each RPC attempt because it may change between queries or retries.
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
    private final KvcmMetricsReporter metricsReporter;
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
    private volatile Consumer<KvcmHealthSnapshot> healthSnapshotListener = ignored -> {
    };

    /**
     * Creates a KVCM client and starts periodic leader and worker-metadata refreshes when KVCM is
     * enabled.
     *
     * <p>The first refresh is scheduled immediately on the dedicated refresh executor. It is not a
     * Spring startup barrier, so unavailable KVCM metadata cannot prevent the application from
     * starting.
     *
     * @param configuration cache-matching configuration
     * @param metaServiceClient low-level KVCM MetaService client
     * @param leaderResolver resolver for the current KVCM leader
     * @param workerMetadataResolver cache for worker-derived namespaces and query types
     * @param applicationWarmupState application warm-up state used for health transitions
     * @param metricsReporter reporter for KVCM query retry metrics
     */
    public KvcmGrpcClient(CacheMatchConfiguration configuration,
                          KvcmMetaServiceClient metaServiceClient,
                          KvcmLeaderResolver leaderResolver,
                          KvcmWorkerMetadataResolver workerMetadataResolver,
                          ApplicationWarmupState applicationWarmupState,
                          KvcmMetricsReporter metricsReporter) {
        this.metaServiceClient = metaServiceClient;
        this.leaderResolver = leaderResolver;
        this.workerMetadataResolver = workerMetadataResolver;
        this.applicationWarmupState = applicationWarmupState;
        this.metricsReporter = metricsReporter;
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

    /**
     * Finds the KVCM prefix-match block count for each matching engine.
     *
     * <p>The returned publisher is cold. On subscription it validates request data, reads cached
     * worker metadata, then queries the current KVCM leader. Missing cached metadata returns an
     * empty map and schedules a coalesced background refresh. KVCM status and protocol failures
     * are propagated as {@link KvcmQueryException} after the configured retries.
     *
     * @param requestId request identifier used for tracing and logs
     * @param blockCacheKeys ordered cache block keys to match
     * @param blockSize cache block size used to form the KVCM instance identifier
     * @param roleType role whose engines are being matched
     * @param group selected worker group, or {@code null} for cross-group routing
     * @return a cold publisher of engine address to matched prefix-block count
     */
    public Mono<Map<String, Integer>> findMatchingEngines(String requestId,
                                                           List<Long> blockCacheKeys,
                                                           long blockSize,
                                                           RoleType roleType,
                                                           String group) {
        return Mono.defer(() -> {
            if (!enabled) {
                log.warn("Skipping KVCM cache query because the KVCM client is disabled, "
                                + "requestId={}, role={}, group={}",
                        requestId, roleType, group);
                return Mono.just(Collections.emptyMap());
            }
            if (blockCacheKeys == null || blockCacheKeys.isEmpty()) {
                log.debug("Skipping KVCM cache query because blockCacheKeys is empty, requestId={}", requestId);
                return Mono.just(Collections.emptyMap());
            }
            if (blockSize <= 0) {
                log.warn("Skipping KVCM cache query because blockSize is unavailable, "
                                + "requestId={}, role={}, group={}",
                        requestId, roleType, group);
                return Mono.just(Collections.emptyMap());
            }

            String namespace = workerMetadataResolver.resolveNamespace(roleType, group, blockSize);
            if (StringUtils.isBlank(namespace)) {
                requestImmediateRefresh();
                log.warn("Skipping KVCM cache query because namespace is unavailable, "
                                + "requestId={}, role={}, group={}",
                        requestId, roleType, group);
                return Mono.just(Collections.emptyMap());
            }
            QueryType queryType = workerMetadataResolver.resolveQueryType(roleType, group);
            if (queryType == null) {
                requestImmediateRefresh();
                log.warn("Skipping KVCM cache query because query type is unavailable, "
                        + "requestId={}, role={}, group={}", requestId, roleType, group);
                return Mono.just(Collections.emptyMap());
            }
            return queryWithRetry(requestId, blockCacheKeys, namespace, queryType, roleType, group, 0);
        });
    }

    private Mono<Map<String, Integer>> queryWithRetry(String requestId,
                                                      List<Long> blockCacheKeys,
                                                      String namespace,
                                                      QueryType queryType,
                                                      RoleType roleType,
                                                      String group,
                                                      int attemptIndex) {
        return queryOnce(requestId, blockCacheKeys, namespace, queryType, roleType, group)
                .doOnSuccess(ignored -> recordQuerySuccess())
                .onErrorResume(failure -> {
                    if (attemptIndex == maxQueryRetryCount) {
                        recordQueryFailure();
                        return Mono.error(failure);
                    }
                    metricsReporter.reportQueryRetry(attemptIndex + 1);
                    log.debug("KVCM cache query failed; retrying, requestId={}, attempt={}, maxRetryCount={}",
                            requestId, attemptIndex + 1, maxQueryRetryCount, failure);
                    return queryWithRetry(
                            requestId, blockCacheKeys, namespace, queryType, roleType, group, attemptIndex + 1);
                });
    }

    private Mono<Map<String, Integer>> queryOnce(String requestId,
                                                 List<Long> blockCacheKeys,
                                                 String namespace,
                                                 QueryType queryType,
                                                 RoleType roleType,
                                                 String group) {
        return Mono.defer(() -> {
            GrpcTarget currentLeader = leaderResolver.resolve();
            if (currentLeader == null) {
                return Mono.error(new KvcmQueryException("KVCM leader is unavailable"));
            }

            String traceId = IdUtils.fastUuid();
            GetHostCacheStateRequest request = GetHostCacheStateRequest.newBuilder()
                    .setTraceId(traceId)
                    // KVCM exposes the cache namespace as instance_id in its protocol.
                    .setInstanceId(namespace)
                    .setQueryType(queryType)
                    .addAllBlockCacheKeys(blockCacheKeys)
                    .build();
            if (log.isDebugEnabled()) {
                log.debug("KVCM GetHostCacheState request: requestId={}, traceId={}, namespace={}, "
                                + "leader={}, role={}, group={}, queryType={}, blockCount={}, blockCacheKeys={}",
                        requestId, traceId, namespace, currentLeader, roleType, group, queryType,
                        blockCacheKeys.size(), blockCacheKeys);
            }
            return metaServiceClient.getHostCacheState(currentLeader, request, config.getRequestTimeoutMs())
                    .publishOn(Schedulers.parallel())
                    .flatMap(response -> mapCacheMatches(requestId, traceId, response));
        }).onErrorMap(StatusRuntimeException.class,
                error -> new KvcmQueryException("KVCM GetHostCacheState gRPC request failed", error));
    }

    private Mono<Map<String, Integer>> mapCacheMatches(
            String requestId,
            String traceId,
            GetHostCacheStateResponse response) {
        ErrorCode code = response.getHeader().getStatus().getCode();
        if (code != ErrorCode.OK) {
            return Mono.error(new KvcmQueryException(
                    "KVCM GetHostCacheState failed, code=" + code + ", message="
                            + response.getHeader().getStatus().getMessage()));
        }
        Map<String, Integer> matches = toPrefixMatchBlocksByHost(response.getHostsList());
        if (log.isDebugEnabled()) {
            log.debug("KVCM GetHostCacheState response: requestId={}, traceId={}, matches={}",
                    requestId, traceId, matches);
        }
        return Mono.just(matches);
    }

    /**
     * Refreshes the leader heartbeat and worker metadata while isolating either refresh failure.
     *
     * <p>This method is invoked by the periodic refresh executor and is package-visible for
     * lifecycle tests.
     */
    void refreshKvcmServiceStateSafely() {
        try {
            recordHeartbeat(leaderResolver.refresh());
        } catch (Exception exception) {
            log.warn("Failed to refresh KVCM leader state; keeping the last known value", exception);
            recordHeartbeat(false);
        }
        try {
            workerMetadataResolver.refreshNamespacesAndQueryTypes();
        } catch (Exception exception) {
            log.warn("Failed to refresh KVCM namespaces and query types; keeping the last known values", exception);
        }
    }

    /**
     * Returns an immutable snapshot of the current KVCM health counters and transition reason.
     *
     * @return current KVCM health snapshot
     */
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

    /**
     * Registers the recipient of KVCM health snapshots.
     *
     * <p>The listener is invoked on an internal refresh or reactive query-completion thread, so it
     * must not block. Exceptions thrown by the listener are logged and do not stop future health
     * refreshes.
     *
     * @param healthSnapshotListener listener that receives evaluated health snapshots
     */
    public void setHealthSnapshotListener(Consumer<KvcmHealthSnapshot> healthSnapshotListener) {
        this.healthSnapshotListener = healthSnapshotListener;
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

    /**
     * Stops background refreshes and releases KVCM gRPC channels.
     */
    @PreDestroy
    public void shutdown() {
        if (refreshExecutor != null) {
            refreshExecutor.shutdown();
        }
        metaServiceClient.shutdown();
    }

}
