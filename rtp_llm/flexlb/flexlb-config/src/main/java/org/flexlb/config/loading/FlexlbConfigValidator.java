package org.flexlb.config.loading;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.cache.CacheMatchingConfig;
import org.flexlb.config.cache.KvcmCacheMatchingConfig;
import org.flexlb.config.cache.LocalStandbyConfig;
import org.flexlb.config.master.ConsistencyConfig;
import org.flexlb.config.master.ZookeeperConsistencyConfig;
import org.flexlb.config.monitoring.NettyMemoryReporterConfig;
import org.flexlb.config.monitoring.ObservabilityConfig;
import org.flexlb.config.routing.OptimizerRuntimeConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.routing.RoutingConfig.CacheAffinityConfig;
import org.flexlb.config.routing.RoutingConfig.DecodeAvailabilityConfig;
import org.flexlb.config.routing.RoutingConfig.DecodeCostEstimatorConfig;
import org.flexlb.config.routing.RoutingConfig.PrefillConfig;
import org.flexlb.config.routing.TrafficPolicyConfig;
import org.flexlb.config.scheduler.PreemptionConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.config.scheduler.QueueOrderingConfig;
import org.flexlb.config.scheduler.RequestLifecycleConfig;
import org.flexlb.config.scheduler.SchedulerConfig;
import org.flexlb.config.worker.WorkerRegistryConfig;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.util.discovery.DiscoveryClientSettings;
import org.flexlb.util.formula.DecodeCostFormula;

/**
 * Cross-field validation for the public configuration contract.
 */
public final class FlexlbConfigValidator {
    private FlexlbConfigValidator() {}

    public static void validateDocumentShape(JsonNode document) {
        validateDocumentShape(document, document);
    }

    public static void validateDocumentShape(JsonNode document, JsonNode effectiveDocument) {
        JsonNode scheduler = document.path("scheduler");
        if (scheduler.isObject()) {
            String type = effectiveDocument.path("scheduler").path("type").asText("QUEUE");
            if ("DIRECT".equals(type)) {
                rejectFieldsExcept(scheduler, "scheduler", "type", "nettyMemoryReporter");
            } else if ("QUEUE".equals(type)) {
                validateOrderingShape(
                        scheduler.path("ordering"),
                        effectiveDocument.path("scheduler").path("ordering"));
            }
        }

        JsonNode decodeCostEstimator =
                document.path("router").path("roles").path("decode").path("costEstimator");
        if (decodeCostEstimator.isObject() && decodeCostEstimator.has("expression")) {
            require(
                    decodeCostEstimator.path("expression").isTextual(),
                    "router.roles.decode.costEstimator.expression",
                    "must be a string");
        }
    }

    public static void validate(FlexlbConfig config) {
        require(
                config.getSchemaVersion() == FlexlbConfig.CURRENT_SCHEMA_VERSION,
                "schemaVersion",
                "must equal " + FlexlbConfig.CURRENT_SCHEMA_VERSION);
        require(config.getScheduler() != null, "scheduler", "is required");
        require(config.getScheduler().getType() != null, "scheduler.type", "is required");
        validateNettyMemoryReporter(config.getScheduler().getNettyMemoryReporter());
        require(config.getRouter() != null, "router", "is required");
        require(config.getWorkerRegistry() != null, "workerRegistry", "is required");
        require(config.getObservability() != null, "observability", "is required");
        require(config.getServiceDiscovery() != null, "serviceDiscovery", "is required");
        require(config.getCacheMatching() != null, "cacheMatching", "is required");
        require(config.getOptimizer() != null, "optimizer", "is required");
        require(config.getConsistency() != null, "consistency", "is required");

        if (!config.isDirect()) {
            validateQueue(config.queueScheduler());
        }
        validateRequestLifecycle(config.getRequestLifecycle());
        validateRouting(config.getRouter());
        validateWorkerRegistry(config.getWorkerRegistry());
        validateObservability(config.getObservability());
        validateServiceDiscovery(config.getServiceDiscovery());
        validateCacheMatching(config.getCacheMatching());
        validateOptimizer(config.getOptimizer());
        validateConsistency(config.getConsistency());
        FlexlbConfig.GrpcServerConfig grpc = config.getGrpcServer();
        require(grpc != null, "grpcServer", "is required");
        nonNegative(grpc.getExecutorCoreSize(), "grpcServer.executorCoreSize");
        positive(grpc.getExecutorMaxSize(), "grpcServer.executorMaxSize");
        positive(grpc.getExecutorQueueSize(), "grpcServer.executorQueueSize");
        nonNegative(grpc.getReceiveBudgetBytes(), "grpcServer.receiveBudgetBytes");
        nonNegative(grpc.getMaxReceivingRequests(), "grpcServer.maxReceivingRequests");
        nonNegative(
                grpc.getMaxReceivingControlRequests(), "grpcServer.maxReceivingControlRequests");
        positive(grpc.getReceiveTimeoutMs(), "grpcServer.receiveTimeoutMs");
        positive(grpc.getShutdownQuietPeriodMs(), "grpcServer.shutdownQuietPeriodMs");
        require(
                grpc.getExecutorMaxSize() >= grpc.getExecutorCoreSize(),
                "grpcServer.executorMaxSize",
                "must be at least executorCoreSize");
    }

    private static void validateOrderingShape(JsonNode ordering, JsonNode effectiveOrdering) {
        if (!ordering.isObject()) {
            return;
        }
        String type = effectiveOrdering.path("type").asText("FIFO");
        if ("FIFO".equals(type)) {
            rejectFieldsExcept(ordering, "scheduler.ordering", "type");
        } else if ("PRIORITY".equals(type)) {
            rejectFieldsExcept(
                    ordering, "scheduler.ordering", "type", "defaultPriority", "preemption");
            validatePreemptionShape(ordering.path("preemption"));
        }
    }

    private static void rejectFieldsExcept(JsonNode object, String path, String... allowed) {
        java.util.Set<String> names = java.util.Set.of(allowed);
        object.fieldNames()
                .forEachRemaining(
                        field -> {
                            if (!names.contains(field)) {
                                throw new ConfigValidationException(
                                        path + "." + field, "is not supported by the active mode");
                            }
                        });
    }

    private static void validatePreemptionShape(JsonNode preemption) {
        if (!preemption.isObject()
                || !preemption.has("timeoutMs")
                || !preemption.has("allowedVictimStages")) {
            return;
        }
        for (JsonNode stage : preemption.path("allowedVictimStages")) {
            if (PreemptionStage.DECODE_ENGINE_OWNED.name().equals(stage.asText())) {
                return;
            }
        }
        throw new ConfigValidationException(
                "scheduler.ordering.preemption.timeoutMs",
                "is supported only when DECODE_ENGINE_OWNED is enabled");
    }

    private static void validateNettyMemoryReporter(NettyMemoryReporterConfig memory) {
        String path = "scheduler.nettyMemoryReporter";
        require(memory != null, path, "is required");
        require(memory.getIntervalMs() >= 1_000L, path + ".intervalMs", "must be at least 1000 ms");
        var capture = memory.getAutoCapture();
        require(capture != null, path + ".autoCapture", "is required");
        positive(capture.getThresholdBytes(), path + ".autoCapture.thresholdBytes");
        require(
                capture.getCooldownMs() >= 1_000L,
                path + ".autoCapture.cooldownMs",
                "must be at least 1000 ms");
        require(
                !capture.isEnabled() || memory.isEnabled(),
                path + ".autoCapture.enabled",
                "requires nettyMemoryReporter.enabled=true");
    }

    private static void require(boolean condition, String field, String message) {
        if (!condition) {
            throw new ConfigValidationException(field, message);
        }
    }

    private static void positive(long value, String field) {
        require(value > 0, field, "must be greater than zero");
    }

    private static void validateQueue(SchedulerConfig queue) {
        positive(queue.getQueueTimeoutMs(), "scheduler.queueTimeoutMs");
        require(
                Double.isFinite(queue.getScanBudgetMultiplier())
                        && queue.getScanBudgetMultiplier() >= 1.0,
                "scheduler.scanBudgetMultiplier",
                "must be finite and at least 1.0");
        require(queue.getOrdering() != null, "scheduler.ordering", "is required for QUEUE");
        QueueOrderingConfig ordering = queue.getOrdering();
        require(ordering.getType() != null, "scheduler.ordering.type", "is required");
        if (ordering.getType() == QueueOrderingConfig.Type.PRIORITY) {
            range(
                    ordering.getDefaultPriority(),
                    PriorityNormalizer.MIN_PRIORITY,
                    PriorityNormalizer.MAX_PRIORITY,
                    "scheduler.ordering.defaultPriority");
            PreemptionConfig preemption = ordering.getPreemption();
            if (preemption != null) {
                require(
                        preemption.getAllowedPreemptionStages() != null
                                && !preemption.getAllowedPreemptionStages().isEmpty(),
                        "scheduler.ordering.preemption.allowedVictimStages",
                        "must contain at least one stage when preemption is configured");
                positive(preemption.getTimeoutMs(), "scheduler.ordering.preemption.timeoutMs");
                require(
                        preemption.getEngineCancellation() != null,
                        "scheduler.ordering.preemption.engineCancellation",
                        "is required");
                require(
                        preemption.getEngineCancellation().getMode() != null,
                        "scheduler.ordering.preemption.engineCancellation.mode",
                        "is required");
            }
        } else {
            require(
                    ordering.getPreemption() == null,
                    "scheduler.ordering.preemption",
                    "is supported only with PRIORITY");
        }
    }

    private static void range(long value, long minimum, long maximum, String field) {
        require(
                value >= minimum && value <= maximum,
                field,
                "must be in [" + minimum + ", " + maximum + "]");
    }

    private static void range(double value, double minimum, double maximum, String field) {
        require(
                Double.isFinite(value) && value >= minimum && value <= maximum,
                field,
                "must be finite and in [" + minimum + ", " + maximum + "]");
    }

    private static void validateRouting(RoutingConfig routing) {
        require(routing.getRoles() != null, "router.roles", "is required");
        require(routing.getRoles().getEncoder() != null, "router.roles.encoder", "is required");
        positive(
                routing.getRoles().getEncoder().getMaxRequests(),
                "router.roles.encoder.maxRequests");
        PrefillConfig prefill = routing.getRoles().getPrefill();
        require(prefill != null, "router.roles.prefill", "is required");
        positive(prefill.getMaxComputeTokens(), "router.roles.prefill.maxComputeTokens");
        try {
            new org.flexlb.util.formula.PrefillTimePredictor(prefill.getCostExpression());
        } catch (IllegalArgumentException error) {
            throw new ConfigValidationException(
                    "router.roles.prefill.costExpression",
                    "contains an invalid formula: " + error.getMessage(),
                    error);
        }
        CacheAffinityConfig affinity = prefill.getCacheAffinity();
        if (affinity != null) {
            if (affinity.getMaxComputeTokens() != null)
                positive(
                        affinity.getMaxComputeTokens(),
                        "router.roles.prefill.cacheAffinity.maxComputeTokens");
            range(
                    affinity.getMinPrefixHitPercent(),
                    0,
                    RoutingConfig.PERCENTAGE_SCALE,
                    "router.roles.prefill.cacheAffinity.minPrefixHitPercent");
            range(
                    affinity.getRemoteDiscount(),
                    0,
                    1,
                    "router.roles.prefill.cacheAffinity.remoteDiscount");
        }

        require(routing.getRoles().getDecode() != null, "router.roles.decode", "is required");
        DecodeCostEstimatorConfig costEstimator = routing.getRoles().getDecode().getCostEstimator();
        require(costEstimator != null, "router.roles.decode.costEstimator", "is required");
        require(
                costEstimator.getExpression() != null && !costEstimator.getExpression().isBlank(),
                "router.roles.decode.costEstimator.expression",
                "must not be blank");
        DecodeCostFormula costFormula;
        try {
            costFormula = costEstimator.compiledFormula();
        } catch (IllegalArgumentException error) {
            throw new ConfigValidationException(
                    "router.roles.decode.costEstimator.expression",
                    "contains an invalid formula: " + error.getMessage(),
                    error);
        }
        DecodeAvailabilityConfig decodeAvailability =
                routing.getRoles().getDecode().getAvailability();
        require(decodeAvailability != null, "router.roles.decode.availability", "is required");
        if (costFormula.requiresMaxRunningSize()) {
            require(
                    decodeAvailability.getMaxEngineRequests() != null,
                    "router.roles.decode.availability.maxEngineRequests",
                    "is required when costEstimator.expression uses max_running_size");
        }
        range(
                decodeAvailability.getMaxKvUsagePercent(),
                1,
                RoutingConfig.PERCENTAGE_SCALE,
                "router.roles.decode.availability.maxKvUsagePercent");
        if (decodeAvailability.getMaxEngineRequests() != null) {
            positive(
                    decodeAvailability.getMaxEngineRequests(),
                    "router.roles.decode.availability.maxEngineRequests");
        }
        if (routing.getGroupSelector() != null) {
            TrafficPolicyConfig.validate(routing.getGroupSelector());
        }
    }

    private static void validateRequestLifecycle(RequestLifecycleConfig lifecycle) {
        require(lifecycle != null, "requestLifecycle", "is required");
        require(lifecycle.getRequest() != null, "requestLifecycle.request", "is required");
        requiredPositive(
                lifecycle.getRequest().getTimeoutMs(), "requestLifecycle.request.timeoutMs");
    }

    private static void requiredPositive(Long value, String field) {
        require(value != null, field, "is required");
        positive(value, field);
    }

    private static void validateWorkerRegistry(WorkerRegistryConfig workers) {
        require(workers.getHealth() != null, "workerRegistry.health", "is required");
        positive(
                workers.getHealth().getStatusPollIntervalMs(),
                "workerRegistry.health.statusPollIntervalMs");
        positive(
                workers.getHealth().getStatusRpcTimeoutMs(),
                "workerRegistry.health.statusRpcTimeoutMs");
        positive(
                workers.getHealth().getCleanupIntervalMs(),
                "workerRegistry.health.cleanupIntervalMs");
        require(workers.getCacheStatus() != null, "workerRegistry.cacheStatus", "is required");
        positive(
                workers.getCacheStatus().getTargetDiffSize(),
                "workerRegistry.cacheStatus.targetDiffSize");
        positive(
                workers.getCacheStatus().getMinRefreshIntervalMs(),
                "workerRegistry.cacheStatus.minRefreshIntervalMs");
        require(
                workers.getCacheStatus().getMaxRefreshIntervalMs()
                        >= workers.getCacheStatus().getMinRefreshIntervalMs(),
                "workerRegistry.cacheStatus.maxRefreshIntervalMs",
                "must be greater than or equal to minRefreshIntervalMs");
    }

    private static void validateObservability(ObservabilityConfig observability) {
        require(observability.getLogging() != null, "observability.logging", "is required");
        require(
                observability.getLogging().getLevel() != null,
                "observability.logging.level",
                "is required");
        require(observability.getCacheHit() != null, "observability.cacheHit", "is required");
        ObservabilityConfig.CacheHitConfig cacheHit = observability.getCacheHit();
        require(
                cacheHit.getRecentKeyWindow() != null,
                "observability.cacheHit.recentKeyWindow",
                "is required");
        positive(
                cacheHit.getRecentKeyWindow().getDurationMs(),
                "observability.cacheHit.recentKeyWindow.durationMs");
        positive(
                cacheHit.getRecentKeyWindow().getMaxKeyOccurrences(),
                "observability.cacheHit.recentKeyWindow.maxKeyOccurrences");
        if (cacheHit.getTheoryLog() != null) {
            require(
                    cacheHit.getTheoryLog().getPath() != null
                            && !cacheHit.getTheoryLog().getPath().isBlank(),
                    "observability.cacheHit.theoryLog.path",
                    "must not be blank");
        }
    }

    private static void validateServiceDiscovery(DiscoveryClientSettings serviceDiscovery) {
        positive(serviceDiscovery.getConnectTimeoutMs(), "serviceDiscovery.connectTimeoutMs");
        positive(serviceDiscovery.getReadTimeoutMs(), "serviceDiscovery.readTimeoutMs");
        positive(serviceDiscovery.getPollIntervalMs(), "serviceDiscovery.pollIntervalMs");
        positive(serviceDiscovery.getMaxIdleConnections(), "serviceDiscovery.maxIdleConnections");
        positive(serviceDiscovery.getKeepAliveDurationMs(), "serviceDiscovery.keepAliveDurationMs");
    }

    private static void validateCacheMatching(CacheMatchingConfig cacheMatching) {
        if (!(cacheMatching instanceof KvcmCacheMatchingConfig kvcm)) {
            return;
        }
        positive(kvcm.getRequestTimeoutMs(), "cacheMatching.requestTimeoutMs");
        positive(kvcm.getLeaderRefreshIntervalMs(), "cacheMatching.leaderRefreshIntervalMs");
        positive(kvcm.getHeartbeatFailureThreshold(), "cacheMatching.heartbeatFailureThreshold");
        positive(kvcm.getQueryFailureThreshold(), "cacheMatching.queryFailureThreshold");
        nonNegative(kvcm.getMaxQueryRetryCount(), "cacheMatching.maxQueryRetryCount");
        positive(kvcm.getRecoverySuccessThreshold(), "cacheMatching.recoverySuccessThreshold");
        nonNegative(kvcm.getGlobalKvsHostCount(), "cacheMatching.globalKvsHostCount");
        require(
                kvcm.getLocalStandby() != null,
                "cacheMatching.localStandby",
                "is required for KVCM");
        validateLocalStandby(kvcm.getLocalStandby());
    }

    private static void nonNegative(long value, String field) {
        require(value >= 0, field, "must be non-negative");
    }

    private static void validateLocalStandby(LocalStandbyConfig localStandby) {
        positive(localStandby.getTtlMs(), "cacheMatching.localStandby.ttlMs");
        positive(localStandby.getMinimumTtlMs(), "cacheMatching.localStandby.minimumTtlMs");
        require(
                localStandby.getMinimumTtlMs() <= localStandby.getTtlMs(),
                "cacheMatching.localStandby.minimumTtlMs",
                "must be less than or equal to ttlMs");
        require(
                Double.isFinite(localStandby.getTtlReductionStartRatio())
                        && localStandby.getTtlReductionStartRatio() >= 0
                        && localStandby.getTtlReductionStartRatio() <= 1,
                "cacheMatching.localStandby.ttlReductionStartRatio",
                "must be finite and in [0, 1]");
        positive(localStandby.getMaximumEntries(), "cacheMatching.localStandby.maximumEntries");
        require(
                Double.isFinite(localStandby.getCapacityMultiplier())
                        && localStandby.getCapacityMultiplier() >= 1,
                "cacheMatching.localStandby.capacityMultiplier",
                "must be finite and greater than or equal to 1");
        positive(
                localStandby.getAsyncQueueCapacity(),
                "cacheMatching.localStandby.asyncQueueCapacity");
    }

    private static void validateOptimizer(OptimizerRuntimeConfig optimizer) {
        positive(optimizer.getDiscoveryPollIntervalMs(), "optimizer.discoveryPollIntervalMs");
    }

    private static void validateConsistency(ConsistencyConfig consistency) {
        if (!(consistency instanceof ZookeeperConsistencyConfig zookeeper)) {
            return;
        }
        require(
                zookeeper.getConnectString() != null && !zookeeper.getConnectString().isBlank(),
                "consistency.connectString",
                "must not be blank for ZOOKEEPER");
        positive(zookeeper.getSessionTimeoutMs(), "consistency.sessionTimeoutMs");
        positive(zookeeper.getConnectionTimeoutMs(), "consistency.connectionTimeoutMs");
        positive(zookeeper.getMasterRefreshIntervalMs(), "consistency.masterRefreshIntervalMs");
    }
}
