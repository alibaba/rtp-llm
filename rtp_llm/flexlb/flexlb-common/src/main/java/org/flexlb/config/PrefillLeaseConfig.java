package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Map;

/**
 * Optional test controller for the number of Prefill workers publishing chunks.
 */
@Getter
@Setter
public final class PrefillLeaseConfig {
    private boolean enabled;
    private boolean automatic;
    private List<String> groups = List.of();
    private int minServingWorkers = 1;
    private int maxServingWorkers = Integer.MAX_VALUE;
    private long evaluationIntervalMs = 10000;
    private long windowSeconds = 60;
    private long settleSeconds = 30;
    private long maxMetricAgeSeconds = 30;
    private long minSamples = 100;
    private int consecutiveViolations = 3;
    private double ttftLowerMs = 100;
    private double ttftUpperMs = 200;
    private String metricStoreUrl = "";
    private String metricToken = "";
    private String metricTenant = "";
    private long queryTimeoutMs = 3000;
    private String ttftMetric = "vllm.time_to_first_token_seconds";
    private String metricNodeLabel = "host.name";
    private Map<String, String> metricLabels = Map.of();
    /**
     * Explicit Master logical address to metric node label; never inferred from readiness.
     */
    private Map<String, String> workerMetricNodes = Map.of();
}
