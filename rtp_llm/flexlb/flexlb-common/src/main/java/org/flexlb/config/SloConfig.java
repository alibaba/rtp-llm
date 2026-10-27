package org.flexlb.config;

import com.fasterxml.jackson.databind.JsonNode;

/** Expand the single user target into internal policies; never silently ignore legacy overrides. */
final class SloConfig {
    static void apply(JsonNode document, FlexlbConfig config) {
        Long slo = config.getTtftSloMs();
        if (slo == null) { return; }
        if (slo <= 0 || slo > Long.MAX_VALUE / 4) {
            throw new ConfigValidationException("ttftSloMs", "must be positive and at most Long.MAX_VALUE / 4");
        }
        reject(document, "/scheduler");
        reject(document, "/dispatcher/maxInflightPerPrefillWorker");
        reject(document, "/router/roles/prefill/cacheAffinity");
        JsonNode balance = document.at("/router/roles/prefill/tokenBalance");
        if (!balance.isMissingNode() && (!balance.isObject() || !balance.isEmpty())) {
            throw new ConfigValidationException("router.roles.prefill.tokenBalance", "is automatic with ttftSloMs");
        }
        var scheduler = config.getScheduler();
        scheduler.setType(SchedulerConfig.Type.QUEUE);
        scheduler.setQueueTimeoutMs(slo * 2);
        var shedding = new PrefillQueueSheddingConfig();
        shedding.setExpectedWaitThresholdMs(slo);
        shedding.setMaxWaitMs(slo * 2);
        scheduler.setPrefillQueueShedding(shedding);
        var decision = scheduler.getDecision();
        boolean batch = config.getDispatcher().getType() == DispatcherConfig.Type.BATCH;
        decision.setType(batch ? DecisionPolicyConfig.Type.FIXED_WINDOW : DecisionPolicyConfig.Type.SINGLE);
        if (batch) {
            // Implementation scan bound, not a worker admission or user capacity limit.
            decision.setMaxRequests(1024);
            decision.setMaxCollectionWaitMs(Math.min(20, slo / 100));
            decision.setMaxPredictedExecutionMs(Math.max(1, slo - slo / 10));
        }
        var health = config.getWorkerRegistry().getHealth();
        if (document.at("/workerRegistry/health").isMissingNode()) {
            // Status freshness and retirement detection must be useful within the user target.
            long poll = Math.clamp(slo / 150, 10, 100);
            long rpc = Math.max(poll * 2, Math.min(1000, slo / 15));
            health.setStatusPollIntervalMs(poll);
            health.setStatusRpcTimeoutMs(rpc);
            health.setStatusStaleAfterMs(Math.max(rpc * 2, Math.min(5000, slo / 6)));
            health.setCleanupIntervalMs(Math.max(poll, Math.min(200, slo / 15)));
        }
        var prefill = config.getRouter().getRoles().getPrefill();
        prefill.setTokenBalance(new RoutingConfig.TokenBalanceConfig());
        if (document.at("/router/roles/prefill/executionTimeEstimator").isMissingNode()) {
            prefill.getExecutionTimeEstimator().setType(RoutingConfig.EstimatorType.LEARNING);
        }
        if (config.getRequestLifecycle().getRequest().getTimeoutMs() == null) {
            // Engine-status inactivity is not the TTFT target or the generation duration.
            config.getRequestLifecycle().getRequest().setTimeoutMs(Math.max(60_000, slo * 4));
        }
    }

    private static void reject(JsonNode document, String path) {
        if (!document.at(path).isMissingNode()) {
            throw new ConfigValidationException(path.substring(1).replace('/', '.'), "is automatic with ttftSloMs; remove the legacy override");
        }
    }

    private SloConfig() { }
}
