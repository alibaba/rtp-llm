package org.flexlb.service.grace.strategy;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.resource.ResourceMeasure;
import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.listener.AppOnlineHooker;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.springframework.stereotype.Component;

import javax.annotation.PreDestroy;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

@Slf4j
@Component
public class QueryWarmerHooker implements AppOnlineHooker {

    public static volatile boolean warmUpFinished;
    private static final long DEFAULT_MAX_WAIT_MS = 300_000L;
    private static final long DEFAULT_CHECK_INTERVAL_MS = 1_000L;
    private static final long DEFAULT_WORKER_STATUS_FRESH_MS = 30_000L;
    private static final int DEFAULT_STABLE_CHECKS = 3;
    private static final long LEGACY_WARMUP_MS = 3_000L;

    private final GracefulLifecycleReporter lifecycleReporter;
    private final ModelMetaConfig modelMetaConfig;
    private final ConfigService configService;
    private final ResourceMeasureFactory resourceMeasureFactory;
    private final long maxWaitMs;
    private final long checkIntervalMs;
    private final long workerStatusFreshMs;
    private final int stableChecks;
    private final AtomicBoolean warmUpInProgress = new AtomicBoolean(false);
    private volatile Thread warmUpThread;

    public QueryWarmerHooker(
            GracefulLifecycleReporter lifecycleReporter,
            ModelMetaConfig modelMetaConfig,
            ConfigService configService,
            ResourceMeasureFactory resourceMeasureFactory) {
        this(
                lifecycleReporter,
                modelMetaConfig,
                configService,
                resourceMeasureFactory,
                readLongEnv("FLEXLB_ONLINE_WARMUP_MAX_WAIT_MS", DEFAULT_MAX_WAIT_MS),
                readLongEnv("FLEXLB_ONLINE_WARMUP_CHECK_INTERVAL_MS", DEFAULT_CHECK_INTERVAL_MS),
                readLongEnv("FLEXLB_ONLINE_WARMUP_WORKER_STATUS_FRESH_MS", DEFAULT_WORKER_STATUS_FRESH_MS),
                readIntEnv("FLEXLB_ONLINE_WARMUP_STABLE_CHECKS", DEFAULT_STABLE_CHECKS)
        );
    }

    QueryWarmerHooker(
            GracefulLifecycleReporter lifecycleReporter,
            ModelMetaConfig modelMetaConfig,
            ConfigService configService,
            ResourceMeasureFactory resourceMeasureFactory,
            long maxWaitMs,
            long checkIntervalMs,
            long workerStatusFreshMs,
            int stableChecks) {
        this.lifecycleReporter = lifecycleReporter;
        this.modelMetaConfig = modelMetaConfig;
        this.configService = configService;
        this.resourceMeasureFactory = resourceMeasureFactory;
        this.maxWaitMs = Math.max(0, maxWaitMs);
        this.checkIntervalMs = Math.max(1, checkIntervalMs);
        this.workerStatusFreshMs = Math.max(0, workerStatusFreshMs);
        this.stableChecks = Math.max(1, stableChecks);
    }

    @Override
    public void afterStartUp() {
        warmUpFinished = false;
        if (!warmUpInProgress.compareAndSet(false, true)) {
            log.info("warm up already in progress");
            return;
        }
        Thread thread = new Thread(this::runWarmUp, "flexlb-route-warmup");
        thread.setDaemon(true);
        warmUpThread = thread;
        thread.start();
    }

    private void runWarmUp() {
        long startTime = System.currentTimeMillis();
        try {
            doWarmUp(startTime);
            long duration = System.currentTimeMillis() - startTime;
            lifecycleReporter.reportWarmerComplete(duration);
            warmUpFinished = true;
            log.info("warm up success, duration={}ms", duration);
        } catch (Exception e) {
            if (Thread.currentThread().isInterrupted()) {
                log.info("warm up interrupted before health online");
            } else {
                log.error("warm up stopped before health online, health remains offline", e);
            }
        } finally {
            warmUpInProgress.set(false);
        }
    }

    @PreDestroy
    public void stopWarmUp() {
        Thread thread = warmUpThread;
        if (thread != null) {
            thread.interrupt();
            try {
                thread.join(Math.min(checkIntervalMs + 100, 1_000L));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @Override
    public int priority() {
        return 0;
    }

    /**
     * Warm up
     */
    private void doWarmUp(long startTimeMs) {
        List<RoleType> requiredRoleTypes = modelMetaConfig.getConfiguredRoleTypes();
        if (requiredRoleTypes.isEmpty()) {
            legacyWarmUp();
            return;
        }

        log.info(
                "do warm up: waiting for route readiness, requiredRoles={}, maxWaitMs={}, checkIntervalMs={}, "
                        + "workerStatusFreshMs={}, stableChecks={}",
                requiredRoleTypes,
                maxWaitMs,
                checkIntervalMs,
                workerStatusFreshMs,
                stableChecks
        );
        int consecutiveReadyChecks = 0;
        long lastTimeoutLogMs = 0;
        while (!Thread.currentThread().isInterrupted()) {
            if (areRequiredRolesReady(requiredRoleTypes)) {
                consecutiveReadyChecks++;
                if (consecutiveReadyChecks >= stableChecks) {
                    return;
                }
            } else {
                consecutiveReadyChecks = 0;
            }
            long now = System.currentTimeMillis();
            if (maxWaitMs > 0 && now - startTimeMs >= maxWaitMs && now - lastTimeoutLogMs >= maxWaitMs) {
                lastTimeoutLogMs = now;
                log.error("warm up still waiting after {}ms, requiredRoles={}", maxWaitMs, requiredRoleTypes);
            }
            sleep(checkIntervalMs);
        }
        throw new IllegalStateException("route warmup interrupted");
    }

    private void legacyWarmUp() {
        log.info("do warm up: no configured service route, using legacy {}ms wait", LEGACY_WARMUP_MS);
        sleep(LEGACY_WARMUP_MS);
    }

    private boolean areRequiredRolesReady(List<RoleType> requiredRoleTypes) {
        Set<String> requiredTrafficGroups = requiredTrafficGroups();
        Set<String> commonRouteableGroups = null;
        for (RoleType roleType : requiredRoleTypes) {
            Map<String, WorkerStatus> roleStatusMap = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getRoleStatusMap(roleType);
            Set<String> roleRouteableGroups = roleStatusMap.values()
                    .stream()
                    .filter(workerStatus -> isRouteableWorker(roleType, workerStatus))
                    .map(workerStatus -> normalizeGroup(workerStatus.getGroup()))
                    .collect(Collectors.toSet());
            if (roleRouteableGroups.isEmpty()) {
                log.info("route warmup waiting for role {}, cachedWorkers={}", roleType, roleStatusMap.size());
                return false;
            }
            if (commonRouteableGroups == null) {
                commonRouteableGroups = new HashSet<>(roleRouteableGroups);
            } else {
                commonRouteableGroups.retainAll(roleRouteableGroups);
                if (commonRouteableGroups.isEmpty()) {
                    log.info("route warmup waiting for common routeable group, role={}, roleGroups={}",
                            roleType, roleRouteableGroups);
                    return false;
                }
            }
        }
        if (commonRouteableGroups == null || commonRouteableGroups.isEmpty()) {
            return false;
        }
        if (requiredTrafficGroups.isEmpty()) {
            return true;
        }
        if (!commonRouteableGroups.containsAll(requiredTrafficGroups)) {
            Set<String> missingGroups = new HashSet<>(requiredTrafficGroups);
            missingGroups.removeAll(commonRouteableGroups);
            log.info(
                    "route warmup waiting for traffic target groups, requiredGroups={}, "
                            + "routeableGroups={}, missingGroups={}",
                    requiredTrafficGroups,
                    commonRouteableGroups,
                    missingGroups
            );
            return false;
        }
        return true;
    }

    private Set<String> requiredTrafficGroups() {
        if (configService == null) {
            return Set.of();
        }
        try {
            FlexlbConfig config = configService.loadBalanceConfig();
            TrafficPolicyConfig trafficPolicy = config == null ? null : config.getTrafficPolicy();
            return trafficPolicy == null ? Set.of() : trafficPolicy.positiveWeightTargetGroups();
        } catch (Exception e) {
            log.warn("load traffic policy target groups failed, skip traffic-group warmup gate", e);
            return Set.of();
        }
    }

    private String normalizeGroup(String group) {
        return group == null ? "" : group;
    }

    private boolean isRouteableWorker(RoleType roleType, WorkerStatus workerStatus) {
        if (!isFreshAliveWorker(workerStatus)) {
            return false;
        }
        if (configService == null || resourceMeasureFactory == null) {
            return workerStatus.getResourceAvailable().get();
        }
        FlexlbConfig config = configService.loadBalanceConfig();
        ResourceMeasure measure = resourceMeasureFactory.getMeasure(config.getResourceMeasureIndicator(roleType));
        return measure == null
                ? workerStatus.getResourceAvailable().get()
                : measure.isResourceAvailable(workerStatus);
    }

    private boolean isFreshAliveWorker(WorkerStatus workerStatus) {
        if (workerStatus == null || !workerStatus.isAlive()) {
            return false;
        }
        if (workerStatusFreshMs <= 0) {
            return true;
        }
        long lastUpdateUs = workerStatus.getStatusLastUpdateTime().get();
        if (lastUpdateUs <= 0) {
            return false;
        }
        long ageUs = System.nanoTime() / 1000 - lastUpdateUs;
        return ageUs <= workerStatusFreshMs * 1000;
    }

    private void sleep(long sleepMs) {
        try {
            Thread.sleep(sleepMs);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("route warmup interrupted", e);
        }
    }

    private static long readLongEnv(String name, long defaultValue) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            return Long.parseLong(raw);
        } catch (NumberFormatException e) {
            log.warn("invalid {}={}, use default {}", name, raw, defaultValue);
            return defaultValue;
        }
    }

    private static int readIntEnv(String name, int defaultValue) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaultValue;
        }
        try {
            return Integer.parseInt(raw);
        } catch (NumberFormatException e) {
            log.warn("invalid {}={}, use default {}", name, raw, defaultValue);
            return defaultValue;
        }
    }
}
