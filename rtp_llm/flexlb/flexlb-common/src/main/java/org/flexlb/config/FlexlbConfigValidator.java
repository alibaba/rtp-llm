package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.balance.prediction.PrefillTimeFormula;
import org.flexlb.config.RoutingConfig.CacheAffinityConfig;
import org.flexlb.config.RoutingConfig.DecodeAvailabilityConfig;
import org.flexlb.config.RoutingConfig.EstimatorType;
import org.flexlb.config.RoutingConfig.PrefillConfig;
import org.flexlb.util.PriorityNormalizer;

/** Cross-field validation for the public configuration contract. */
final class FlexlbConfigValidator {

    private static final int MIN_STALE_TIMEOUT_TO_RPC_TIMEOUT_RATIO = 2;

    static void validateDocumentShape(JsonNode document) {
        JsonNode scheduler = document.path("scheduler");
        if (scheduler.isObject()) {
            String type = scheduler.path("type").asText("QUEUE");
            if ("DIRECT".equals(type)) {
                rejectFieldsExcept(scheduler, "scheduler", "type");
            } else if ("QUEUE".equals(type)) {
                validateOrderingShape(scheduler.path("ordering"));
                validateDecisionShape(scheduler.path("decision"));
            }
        }

        JsonNode dispatcher = document.path("dispatcher");
        if (dispatcher.isObject()) {
            rejectFieldsExcept(dispatcher, "dispatcher", "type", "maxInflightPerPrefillWorker");
        }

        JsonNode prefill = document.path("router").path("roles").path("prefill");
        if (prefill.isObject()) {
            validateEstimatorShape(prefill.path("executionTimeEstimator"));
        }
    }

    private static void validateEstimatorShape(JsonNode estimator) {
        if (!estimator.isObject()) {
            return;
        }
        String type = estimator.path("type").asText("FORMULA");
        if ("FORMULA".equals(type)) {
            rejectFieldsExcept(estimator,
                    "router.roles.prefill.executionTimeEstimator", "type", "expression");
        } else if ("LEARNING".equals(type)) {
            rejectFieldsExcept(estimator,
                    "router.roles.prefill.executionTimeEstimator", "type");
        }
    }

    private static void validateOrderingShape(JsonNode ordering) {
        if (!ordering.isObject()) {
            return;
        }
        String type = ordering.path("type").asText("FIFO");
        if ("FIFO".equals(type)) {
            rejectFieldsExcept(ordering, "scheduler.ordering", "type");
        } else if ("PRIORITY".equals(type)) {
            rejectFieldsExcept(ordering, "scheduler.ordering", "type",
                    "defaultPriority", "preemption");
            validatePreemptionShape(ordering.path("preemption"));
        }
    }

    private static void validatePreemptionShape(JsonNode preemption) {
        if (!preemption.isObject() || !preemption.has("timeoutMs") || !preemption.has("allowedVictimStages")) {
            return;
        }
        for (JsonNode stage : preemption.path("allowedVictimStages")) {
            if (VictimStage.DECODE_ENGINE_OWNED.name().equals(stage.asText())) {
                return;
            }
        }
        throw new ConfigValidationException(
                "scheduler.ordering.preemption.timeoutMs",
                "is supported only when DECODE_ENGINE_OWNED is enabled");
    }

    private static void validateDecisionShape(JsonNode decision) {
        if (!decision.isObject()) {
            return;
        }
        String type = decision.path("type").asText("FIXED_WINDOW");
        if ("SINGLE".equals(type)) {
            rejectFieldsExcept(decision, "scheduler.decision", "type");
        } else if ("FIXED_WINDOW".equals(type)) {
            rejectFieldsExcept(decision, "scheduler.decision", "type",
                    "maxRequests", "maxCollectionWaitMs",
                    "maxPredictedExecutionMs");
        }
    }

    private static void rejectFieldsExcept(
            JsonNode object, String path, String... allowed) {
        java.util.Set<String> names = java.util.Set.of(allowed);
        object.fieldNames().forEachRemaining(field -> {
            if (!names.contains(field)) {
                throw new ConfigValidationException(
                        path + "." + field,
                        "is not supported by the active mode");
            }
        });
    }

    static void validate(FlexlbConfig config) {
        require(config.getSchemaVersion() == FlexlbConfig.CURRENT_SCHEMA_VERSION,
                "schemaVersion", "must equal " + FlexlbConfig.CURRENT_SCHEMA_VERSION);
        require(config.getScheduler() != null, "scheduler", "is required");
        require(config.getScheduler().getType() != null, "scheduler.type", "is required");
        require(config.getDispatcher() != null, "dispatcher", "is required");
        require(config.getRouter() != null, "router", "is required");
        require(config.getWorkerRegistry() != null, "workerRegistry", "is required");
        require(config.getObservability() != null, "observability", "is required");

        if (!config.isDirect()) {
            validateQueue(config.queueScheduler());
        }
        config.getDispatcher().validateFor(config.getScheduler());
        validateRequestLifecycle(config.getRequestLifecycle());
        validateRouting(config.getRouter());
        validateWorkerRegistry(config.getWorkerRegistry());
        validateObservability(config.getObservability());
        FlexlbConfig.GrpcServerConfig grpc = config.getGrpcServer();
        require(grpc != null, "grpcServer", "is required");
        nonNegative(grpc.getExecutorCoreSize(), "grpcServer.executorCoreSize");
        positive(grpc.getExecutorMaxSize(), "grpcServer.executorMaxSize");
        positive(grpc.getExecutorQueueSize(), "grpcServer.executorQueueSize");
        require(grpc.getExecutorMaxSize() >= grpc.getExecutorCoreSize(),
                "grpcServer.executorMaxSize", "must be at least executorCoreSize");
    }

    private static void validateQueue(SchedulerConfig queue) {
        positive(queue.getQueueTimeoutMs(), "scheduler.queueTimeoutMs");
        require(queue.getOrdering() != null, "scheduler.ordering", "is required for QUEUE");
        require(queue.getDecision() != null, "scheduler.decision", "is required for QUEUE");
        DecisionPolicyConfig decision = queue.getDecision();
        require(decision.getType() != null, "scheduler.decision.type", "is required");
        if (decision.getType() == DecisionPolicyConfig.Type.FIXED_WINDOW) {
            positive(decision.getMaxRequests(),
                    "scheduler.decision.maxRequests");
            nonNegative(decision.getMaxCollectionWaitMs(),
                    "scheduler.decision.maxCollectionWaitMs");
            if (decision.getMaxPredictedExecutionMs() != null) {
                positive(decision.getMaxPredictedExecutionMs(),
                        "scheduler.decision.maxPredictedExecutionMs");
            }
        } else {
            require(decision.getMaxRequests() == DecisionPolicyConfig.DEFAULT_MAX_REQUESTS
                            && decision.getMaxCollectionWaitMs()
                            == DecisionPolicyConfig.DEFAULT_MAX_COLLECTION_WAIT_MS
                            && decision.getMaxPredictedExecutionMs() == null,
                    "scheduler.decision",
                    "fixed-window fields are supported only with FIXED_WINDOW");
        }
        QueueOrderingConfig ordering = queue.getOrdering();
        require(ordering.getType() != null, "scheduler.ordering.type", "is required");
        if (ordering.getType() == QueueOrderingConfig.Type.PRIORITY) {
            range(ordering.getDefaultPriority(),
                    PriorityNormalizer.MIN_PRIORITY,
                    PriorityNormalizer.MAX_PRIORITY,
                    "scheduler.ordering.defaultPriority");
            PreemptionConfig preemption = ordering.getPreemption();
            if (preemption != null) {
                require(preemption.getAllowedVictimStages() != null
                                && !preemption.getAllowedVictimStages().isEmpty(),
                        "scheduler.ordering.preemption.allowedVictimStages",
                        "must contain at least one stage when preemption is configured");
                positive(preemption.getTimeoutMs(),
                        "scheduler.ordering.preemption.timeoutMs");
            }
        } else {
            require(ordering.getPreemption() == null,
                    "scheduler.ordering.preemption",
                    "is supported only with PRIORITY");
        }
    }

    private static void validateRouting(RoutingConfig routing) {
        require(routing.getRoles() != null, "router.roles", "is required");
        PrefillConfig prefill = routing.getRoles().getPrefill();
        require(prefill != null, "router.roles.prefill", "is required");
        require(prefill.getExecutionTimeEstimator() != null,
                "router.roles.prefill.executionTimeEstimator", "is required");
        require(prefill.getExecutionTimeEstimator().getType() != null,
                "router.roles.prefill.executionTimeEstimator.type", "is required");
        if (prefill.getExecutionTimeEstimator().getType()
                == EstimatorType.FORMULA) {
            String expression = prefill.getExecutionTimeEstimator().getExpression();
            require(expression != null && !expression.isBlank(),
                    "router.roles.prefill.executionTimeEstimator.expression", "must not be blank");
            try {
                PrefillTimeFormula.parse(expression);
            } catch (IllegalArgumentException error) {
                throw new ConfigValidationException(
                        "router.roles.prefill.executionTimeEstimator.expression",
                        "contains an invalid formula: " + error.getMessage(), error);
            }
        }
        CacheAffinityConfig affinity = prefill.getCacheAffinity();
        if (affinity != null) {
            nonNegative(affinity.getMaxExtraTtftMs(),
                    "router.roles.prefill.cacheAffinity.maxExtraTtftMs");
            range(affinity.getMinPrefixHitPercent(), 0,
                    RoutingConfig.PERCENTAGE_SCALE,
                    "router.roles.prefill.cacheAffinity.minPrefixHitPercent");
        }

        require(routing.getRoles().getDecode() != null,
                "router.roles.decode", "is required");
        DecodeAvailabilityConfig decodeAvailability =
                routing.getRoles().getDecode().getAvailability();
        require(decodeAvailability != null,
                "router.roles.decode.availability", "is required");
        range(decodeAvailability.getMaxKvUsagePercent(), 0,
                RoutingConfig.PERCENTAGE_SCALE,
                "router.roles.decode.availability.maxKvUsagePercent");
        if (decodeAvailability.getMaxEngineRequests() != null) {
            positive(decodeAvailability.getMaxEngineRequests(),
                    "router.roles.decode.availability.maxEngineRequests");
        }
        if (routing.getGroupSelector() != null) {
            TrafficPolicyConfig.validate(routing.getGroupSelector());
        }
    }

    private static void validateRequestLifecycle(RequestLifecycleConfig lifecycle) {
        require(lifecycle != null, "requestLifecycle", "is required");
        require(lifecycle.getRequest() != null, "requestLifecycle.request", "is required");
        requiredPositive(lifecycle.getRequest().getTimeoutMs(), "requestLifecycle.request.timeoutMs");
        require(lifecycle.getDecision() != null, "requestLifecycle.decision", "is required");
        Double lifetime = lifecycle.getDecision().getLifetime();
        require(lifetime != null, "requestLifecycle.decision.lifetime", "is required");
        require(Double.isFinite(lifetime) && lifetime >= 1.0,
                "requestLifecycle.decision.lifetime", "must be finite and at least 1");
    }

    private static void requiredPositive(Long value, String field) {
        require(value != null, field, "is required");
        positive(value, field);
    }

    private static void validateWorkerRegistry(WorkerRegistryConfig workers) {
        require(workers.getHealth() != null, "workerRegistry.health", "is required");
        positive(workers.getHealth().getStatusPollIntervalMs(),
                "workerRegistry.health.statusPollIntervalMs");
        positive(workers.getHealth().getStatusRpcTimeoutMs(),
                "workerRegistry.health.statusRpcTimeoutMs");
        positive(workers.getHealth().getCleanupIntervalMs(),
                "workerRegistry.health.cleanupIntervalMs");
        require(workers.getHealth().getStatusRpcTimeoutMs()
                        <= workers.getHealth().getStatusStaleAfterMs()
                                / MIN_STALE_TIMEOUT_TO_RPC_TIMEOUT_RATIO,
                "workerRegistry.health.statusStaleAfterMs",
                "must be at least twice statusRpcTimeoutMs");
        require(workers.getCacheStatus() != null,
                "workerRegistry.cacheStatus", "is required");
        positive(workers.getCacheStatus().getTargetDiffSize(),
                "workerRegistry.cacheStatus.targetDiffSize");
        positive(workers.getCacheStatus().getMinRefreshIntervalMs(),
                "workerRegistry.cacheStatus.minRefreshIntervalMs");
        require(workers.getCacheStatus().getMaxRefreshIntervalMs()
                        >= workers.getCacheStatus().getMinRefreshIntervalMs(),
                "workerRegistry.cacheStatus.maxRefreshIntervalMs",
                "must be greater than or equal to minRefreshIntervalMs");
    }

    private static void validateObservability(ObservabilityConfig observability) {
        require(observability.getCacheHit() != null,
                "observability.cacheHit", "is required");
        ObservabilityConfig.CacheHitConfig cacheHit = observability.getCacheHit();
        require(cacheHit.getRecentKeyWindow() != null,
                "observability.cacheHit.recentKeyWindow", "is required");
        positive(cacheHit.getRecentKeyWindow().getDurationMs(),
                "observability.cacheHit.recentKeyWindow.durationMs");
        positive(cacheHit.getRecentKeyWindow().getMaxKeyOccurrences(),
                "observability.cacheHit.recentKeyWindow.maxKeyOccurrences");
        if (cacheHit.getTheoryLog() != null) {
            require(cacheHit.getTheoryLog().getPath() != null
                            && !cacheHit.getTheoryLog().getPath().isBlank(),
                    "observability.cacheHit.theoryLog.path", "must not be blank");
        }
    }

    private static void positive(long value, String field) {
        require(value > 0, field, "must be greater than zero");
    }

    private static void positive(double value, String field) {
        require(Double.isFinite(value) && value > 0, field,
                "must be finite and greater than zero");
    }

    private static void nonNegative(long value, String field) {
        require(value >= 0, field, "must be non-negative");
    }

    private static void nonNegative(double value, String field) {
        require(Double.isFinite(value) && value >= 0, field,
                "must be finite and non-negative");
    }

    private static void range(long value, long minimum, long maximum, String field) {
        require(value >= minimum && value <= maximum, field,
                "must be in [" + minimum + ", " + maximum + "]");
    }

    private static void range(double value, double minimum, double maximum, String field) {
        require(Double.isFinite(value) && value >= minimum && value <= maximum, field,
                "must be finite and in [" + minimum + ", " + maximum + "]");
    }

    private static void require(boolean condition, String field, String message) {
        if (!condition) {
            throw new ConfigValidationException(field, message);
        }
    }

    private FlexlbConfigValidator() {
    }
}
