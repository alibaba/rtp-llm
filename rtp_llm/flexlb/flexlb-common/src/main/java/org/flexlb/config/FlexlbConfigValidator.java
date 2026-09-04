package org.flexlb.config;

import org.flexlb.balance.strategy.PrefillTimeFormula;
import org.flexlb.config.RoutingConfig.BestOnlyConfig;
import org.flexlb.config.RoutingConfig.CacheAffinityConfig;
import org.flexlb.config.RoutingConfig.DecodeAvailabilityConfig;
import org.flexlb.config.RoutingConfig.EstimatedTtftSelectorConfig;
import org.flexlb.config.RoutingConfig.FixedCandidatePoolConfig;
import org.flexlb.config.RoutingConfig.FormulaEstimatorConfig;
import org.flexlb.config.RoutingConfig.KvUsageWeightedRandomConfig;
import org.flexlb.config.RoutingConfig.LeastRecentlyUsedInPoolConfig;
import org.flexlb.config.RoutingConfig.OutlierRejectionConfig;
import org.flexlb.config.RoutingConfig.PrefillConfig;
import org.flexlb.config.RoutingConfig.RandomWithinToleranceConfig;
import org.flexlb.config.RoutingConfig.RatioCandidatePoolConfig;

/** Cross-field validation for the public configuration contract. */
final class FlexlbConfigValidator {

    static void validate(FlexlbConfig config) {
        require(config.getSchemaVersion() == FlexlbConfig.CURRENT_SCHEMA_VERSION,
                "schemaVersion", "must equal " + FlexlbConfig.CURRENT_SCHEMA_VERSION);
        require(config.getScheduler() != null, "scheduler", "is required");
        require(config.getDispatcher() != null, "dispatcher", "is required");
        require(config.getRouter() != null, "router", "is required");
        require(config.getWorkerRegistry() != null, "workerRegistry", "is required");
        require(config.getObservability() != null, "observability", "is required");

        if (config.isDirect()) {
            require(config.getDispatcher() instanceof NonBatchDispatcherConfig,
                    "dispatcher.type", "DIRECT requires NON_BATCH");
        } else {
            validateQueue(config, config.queueScheduler());
        }
        validateDispatcher(config);
        validateRouting(config.getRouter());
        validateWorkerRegistry(config.getWorkerRegistry());
        validateObservability(config.getObservability());
    }

    private static void validateQueue(FlexlbConfig config, QueueSchedulerConfig queue) {
        positive(queue.getQueueTimeoutMs(), "scheduler.queueTimeoutMs");
        require(queue.getOrdering() != null, "scheduler.ordering", "is required for QUEUE");
        require(queue.getCapacity() != null, "scheduler.capacity", "is required for QUEUE");
        require(queue.getLifecycle() != null, "scheduler.lifecycle", "is required for QUEUE");
        positive(queue.getCapacity().getMaxOutstandingRequestsGlobal(),
                "scheduler.capacity.maxOutstandingRequestsGlobal");
        positive(queue.getLifecycle().getStaleInflightTimeoutMs(),
                "scheduler.lifecycle.staleInflightTimeoutMs");
        positive(queue.getLifecycle().getDeliveredNotAcceptedTimeoutMs(),
                "scheduler.lifecycle.deliveredNotAcceptedTimeoutMs");
        positive(queue.getLifecycle().getMaxDeliveredNotAcceptedRequestsGlobal(),
                "scheduler.lifecycle.maxDeliveredNotAcceptedRequestsGlobal");
        if (queue.getOrdering() instanceof PriorityOrderingConfig priority) {
            range(priority.getDefaultPriority(), 1, 100,
                    "scheduler.ordering.defaultPriority");
            PreemptionConfig preemption = priority.getPreemption();
            if (preemption != null) {
                require(preemption.getAllowedVictimStages() != null
                                && !preemption.getAllowedVictimStages().isEmpty(),
                        "scheduler.ordering.preemption.allowedVictimStages",
                        "must contain at least one stage when preemption is configured");
                boolean cancelsEngineOwned = preemption.getAllowedVictimStages()
                        .contains(VictimStage.DECODE_ENGINE_OWNED);
                EngineCancellationConfig cancellation = preemption.getEngineCancellation();
                if (cancelsEngineOwned) {
                    require(cancellation != null,
                            "scheduler.ordering.preemption.engineCancellation",
                            "is required when DECODE_ENGINE_OWNED is allowed");
                    positive(cancellation.getAckTimeoutMs(),
                            "scheduler.ordering.preemption.engineCancellation.ackTimeoutMs");
                    positive(cancellation.getCompletionTimeoutMs(),
                            "scheduler.ordering.preemption.engineCancellation.completionTimeoutMs");
                } else {
                    require(cancellation == null,
                            "scheduler.ordering.preemption.engineCancellation",
                            "is allowed only when DECODE_ENGINE_OWNED is allowed");
                }
            }
        }
    }

    private static void validateDispatcher(FlexlbConfig config) {
        DispatcherConfig dispatcher = config.getDispatcher();
        if (dispatcher instanceof BatchDispatcherConfig batch) {
            positive(batch.getMaxRequests(), "dispatcher.maxRequests");
            nonNegative(batch.getMaxCollectionWaitMs(), "dispatcher.maxCollectionWaitMs");
            positive(batch.getMaxWaitingRequestsPerPrefillWorker(),
                    "dispatcher.maxWaitingRequestsPerPrefillWorker");
            positive(batch.getEnqueueRpcTimeoutMs(), "dispatcher.enqueueRpcTimeoutMs");
            if (batch.getEarlyDispatchPredictedExecutionMs() != null) {
                positive(batch.getEarlyDispatchPredictedExecutionMs(),
                        "dispatcher.earlyDispatchPredictedExecutionMs");
            }
            if (batch.getMaxInflightBatchesPerPrefillWorker() != null) {
                positive(batch.getMaxInflightBatchesPerPrefillWorker(),
                        "dispatcher.maxInflightBatchesPerPrefillWorker");
            }
        } else {
            Integer maximum = ((NonBatchDispatcherConfig) dispatcher)
                    .getMaxInflightRequestsPerPrefillWorker();
            if (maximum != null) {
                require(config.isQueue(),
                        "dispatcher.maxInflightRequestsPerPrefillWorker",
                        "is supported only with QUEUE");
                positive(maximum, "dispatcher.maxInflightRequestsPerPrefillWorker");
            }
        }
    }

    private static void validateRouting(RoutingConfig routing) {
        range(routing.getAvailabilityHysteresisPercent(), 0, 100,
                "router.availabilityHysteresisPercent");
        require(routing.getRoles() != null, "router.roles", "is required");
        PrefillConfig prefill = routing.getRoles().getPrefill();
        require(prefill != null, "router.roles.prefill", "is required");
        require(prefill.getAvailability() != null,
                "router.roles.prefill.availability", "is required");
        positive(prefill.getAvailability().getMaxPendingRequests(),
                "router.roles.prefill.availability.maxPendingRequests");
        require(prefill.getExecutionTimeEstimator() != null,
                "router.roles.prefill.executionTimeEstimator", "is required");
        if (prefill.getExecutionTimeEstimator() instanceof FormulaEstimatorConfig formula) {
            require(formula.getExpression() != null && !formula.getExpression().isBlank(),
                    "router.roles.prefill.executionTimeEstimator.expression", "must not be blank");
            try {
                PrefillTimeFormula.parse(formula.getExpression());
            } catch (IllegalArgumentException error) {
                throw new ConfigValidationException(
                        "router.roles.prefill.executionTimeEstimator.expression",
                        "contains an invalid formula: " + error.getMessage(), error);
            }
        }
        require(prefill.getSelector() != null,
                "router.roles.prefill.selector", "is required");
        if (prefill.getSelector() instanceof EstimatedTtftSelectorConfig estimated) {
            require(estimated.getCandidateChoice() != null,
                    "router.roles.prefill.selector.candidateChoice", "is required");
            if (estimated.getCandidateChoice() instanceof RandomWithinToleranceConfig random) {
                validatePrefillOutlierRejection(random.getOutlierRejection());
                range(random.getRelativeTolerance(), 0, 1,
                        "router.roles.prefill.selector.candidateChoice.relativeTolerance");
                nonNegative(random.getMinimumToleranceMs(),
                        "router.roles.prefill.selector.candidateChoice.minimumToleranceMs");
            } else if (estimated.getCandidateChoice() instanceof BestOnlyConfig best) {
                validatePrefillOutlierRejection(best.getOutlierRejection());
            } else if (estimated.getCandidateChoice() instanceof LeastRecentlyUsedInPoolConfig lru) {
                require(lru.getPool() != null,
                        "router.roles.prefill.selector.candidateChoice.pool", "is required");
                if (lru.getPool() instanceof RatioCandidatePoolConfig ratio) {
                    require(ratio.getRatio() > 0 && ratio.getRatio() <= 1,
                            "router.roles.prefill.selector.candidateChoice.pool.ratio",
                            "must be in (0, 1]");
                    positive(ratio.getMinimumWorkers(),
                            "router.roles.prefill.selector.candidateChoice.pool.minimumWorkers");
                } else {
                    positive(((FixedCandidatePoolConfig) lru.getPool()).getWorkers(),
                            "router.roles.prefill.selector.candidateChoice.pool.workers");
                }
            }
        }
        CacheAffinityConfig affinity = prefill.getCacheAffinity();
        if (affinity != null) {
            require(prefill.getSelector() instanceof EstimatedTtftSelectorConfig,
                    "router.roles.prefill.cacheAffinity",
                    "is supported only by the ESTIMATED_TTFT selector");
            nonNegative(affinity.getMaxExtraTtftMs(),
                    "router.roles.prefill.cacheAffinity.maxExtraTtftMs");
            range(affinity.getMinPrefixHitPercent(), 0, 100,
                    "router.roles.prefill.cacheAffinity.minPrefixHitPercent");
        }
        var sessionAffinity = prefill.getSessionAffinity();
        if (sessionAffinity != null) {
            require(prefill.getSelector() instanceof EstimatedTtftSelectorConfig,
                    "router.roles.prefill.sessionAffinity",
                    "is supported only by the ESTIMATED_TTFT selector");
            range(sessionAffinity.getTtlMs(), 1,
                    RoutingConfig.SessionAffinityConfig.MAX_TTL_MS,
                    "router.roles.prefill.sessionAffinity.ttlMs");
            nonNegative(sessionAffinity.getMaxExtraTtftMs(),
                    "router.roles.prefill.sessionAffinity.maxExtraTtftMs");
            positive(sessionAffinity.getMaxEntries(),
                    "router.roles.prefill.sessionAffinity.maxEntries");
        }

        require(routing.getRoles().getDecode() != null,
                "router.roles.decode", "is required");
        DecodeAvailabilityConfig decodeAvailability =
                routing.getRoles().getDecode().getAvailability();
        require(decodeAvailability != null,
                "router.roles.decode.availability", "is required");
        range(decodeAvailability.getMaxKvUsagePercent(), 0, 100,
                "router.roles.decode.availability.maxKvUsagePercent");
        if (decodeAvailability.getMaxEngineRequests() != null) {
            positive(decodeAvailability.getMaxEngineRequests(),
                    "router.roles.decode.availability.maxEngineRequests");
        }
        require(routing.getRoles().getDecode().getKvReservation() != null,
                "router.roles.decode.kvReservation", "is required");
        Long maxOutput = routing.getRoles().getDecode().getKvReservation()
                .getMaxOutputTokensForEstimate();
        if (maxOutput != null) {
            positive(maxOutput, "router.roles.decode.kvReservation.maxOutputTokensForEstimate");
        }
        require(routing.getRoles().getDecode().getSelector() != null,
                "router.roles.decode.selector", "is required");
        if (routing.getRoles().getDecode().getSelector()
                instanceof KvUsageWeightedRandomConfig weighted) {
            nonNegative(weighted.getDecayPerToken(),
                    "router.roles.decode.selector.decayPerToken");
            if (weighted.getOutlierRejection() != null) {
                positive(weighted.getOutlierRejection().getMaxEngineLoadVsAverageMultiplier(),
                        "router.roles.decode.selector.outlierRejection.maxEngineLoadVsAverageMultiplier");
                positive(weighted.getOutlierRejection().getMaxKvUsedVsAverageMultiplier(),
                        "router.roles.decode.selector.outlierRejection.maxKvUsedVsAverageMultiplier");
            }
        }
        require(routing.getRoles().getVit() != null
                        && routing.getRoles().getVit().getSelector() != null,
                "router.roles.vit.selector", "is required");
        if (routing.getGroupSelector() != null) {
            TrafficPolicyConfig.validate(routing.getGroupSelector());
        }
    }

    private static void validatePrefillOutlierRejection(
            OutlierRejectionConfig outlierRejection) {
        if (outlierRejection == null) {
            return;
        }
        positive(outlierRejection.getMaxPendingVsAverageMultiplier(),
                "router.roles.prefill.selector.candidateChoice.outlierRejection"
                        + ".maxPendingVsAverageMultiplier");
        positive(outlierRejection.getMaxWaitVsAverageMultiplier(),
                "router.roles.prefill.selector.candidateChoice.outlierRejection"
                        + ".maxWaitVsAverageMultiplier");
    }

    private static void validateWorkerRegistry(WorkerRegistryConfig workers) {
        require(workers.getHealth() != null, "workerRegistry.health", "is required");
        positive(workers.getHealth().getStatusPollIntervalMs(),
                "workerRegistry.health.statusPollIntervalMs");
        positive(workers.getHealth().getStatusRpcTimeoutMs(),
                "workerRegistry.health.statusRpcTimeoutMs");
        require(workers.getHealth().getStatusRpcTimeoutMs()
                        <= workers.getHealth().getStatusStaleAfterMs() / 2,
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
