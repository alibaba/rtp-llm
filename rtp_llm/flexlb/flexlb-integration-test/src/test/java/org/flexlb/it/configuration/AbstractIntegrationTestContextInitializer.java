package org.flexlb.it.configuration;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.it.fixture.engine.WorkerTopology;
import org.flexlb.it.fixture.kvcm.KvcmIntegrationTestFixtures;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Shared Spring-context assembly for role-aware FlexLB integration-test initializers.
 *
 * <p>Concrete initializers declare their scenario's topology and policy. This class starts shared
 * fake boundaries and registers the matching test-only configuration before production beans are
 * created.
 */
public abstract class AbstractIntegrationTestContextInitializer {

    protected final void initializeShortestTtftContext(WorkerTopology topology) {
        initializeShortestTtftContext(topology, 1);
    }

    /** Starts a direct SHORTEST_TTFT context with an explicit schedule-worker count. */
    protected final void initializeShortestTtftContext(WorkerTopology topology, int scheduleWorkerSize) {
        initializeContext(topology, "SHORTEST_TTFT", false, false, EngineMode.RTP_LLM, "", scheduleWorkerSize);
    }

    protected final void initializeQueueingShortestTtftContext(WorkerTopology topology) {
        initializeContext(topology, "SHORTEST_TTFT", false, true, EngineMode.RTP_LLM, "", 1);
    }

    protected final void initializeFallbackContext(WorkerTopology topology) {
        initializeContext(topology, "SHORTEST_TTFT", true, false, EngineMode.RTP_LLM, "", 1);
    }

    protected final void initializeCacheAffinityContext(WorkerTopology topology, EngineMode engineMode) {
        initializeCacheAffinityContext(topology, engineMode, 60_000, 10_000, 10_000, 2);
    }

    /** Starts a KVCM-backed cache-affinity context with explicit Local Standby retention limits. */
    protected final void initializeCacheAffinityContext(
            WorkerTopology topology,
            EngineMode engineMode,
            long localStandbyTtlMs,
            long localStandbyMinimumTtlMs,
            long localStandbyMaximumEntries,
            int kvcmQueryFailureThreshold) {
        if (topology.workerCount(RoleType.PDFUSION) < 2) {
            throw new IllegalArgumentException("CACHE_AFFINITY_FIRST IT requires at least two PDFUSION workers");
        }
        IntegrationTestFixtures.startWorkers(topology);
        String kvcm = "";
        if (engineMode.usesKvcm()) {
            int kvcmPort = KvcmIntegrationTestFixtures.startKvcm();
            KvcmIntegrationTestFixtures.setMatchingWorker(RoleType.PDFUSION, 1);
            kvcm = kvcmConfig(
                    kvcmPort,
                    localStandbyTtlMs,
                    localStandbyMinimumTtlMs,
                    localStandbyMaximumEntries,
                    kvcmQueryFailureThreshold);
        }
        ConfigService.register(new StaticConfigSource(config(
                "CACHE_AFFINITY_FIRST",
                false,
                false,
                engineMode,
                kvcm,
                1)));
    }

    private void initializeContext(
            WorkerTopology topology,
            String strategy,
            boolean enableFallback,
            boolean enableQueueing,
            EngineMode engineMode,
            String kvcm,
            int scheduleWorkerSize) {
        IntegrationTestFixtures.startWorkers(topology);
        ConfigService.register(new StaticConfigSource(config(
                strategy,
                enableFallback,
                enableQueueing,
                engineMode,
                kvcm,
                scheduleWorkerSize)));
    }

    private String config(
            String strategy,
            boolean enableFallback,
            boolean enableQueueing,
            EngineMode engineMode,
            String kvcm,
            int scheduleWorkerSize) {
        return """
                {
                  "loadBalanceStrategy": "%s",
                  "blockHashStrategy": "%s",
                  "enableFallback": %b,
                  "enableQueueing": %b,
                  "syncStatusInterval": 50,
                  "syncRequestTimeoutMs": 500,
                  "prefillQueueSizeThreshold": 100,
                  "scheduleWorkerSize": %d,
                  "fixedScheduleWorkerPermits": true,
                  "cacheAffinityFirstMaxExtraWorkTokens": 1000000,
                  "cacheAffinityFirstMinHitRate": 0,
                  "modelServiceConfig": {
                    "service_id": "aigc.text-generation.generation.engine_service",
                    "role_endpoints": [
                      {
                        "group": "default",
                        %s
                      }
                    ]%s
                  }
                }
                """.formatted(
                strategy,
                engineMode.blockHashStrategy(),
                enableFallback,
                enableQueueing,
                scheduleWorkerSize,
                roleEndpointsConfig(),
                kvcm);
    }

    private String roleEndpointsConfig() {
        return Arrays.stream(RoleType.values())
                .filter(roleType -> IntegrationTestFixtures.workerCount(roleType) > 0)
                .map(roleType -> "\"%s\": {\n%s\n}".formatted(
                        endpointProperty(roleType), endpointConfig(roleType)))
                .collect(Collectors.joining(",\n"));
    }

    private String endpointConfig(RoleType roleType) {
        return """
                "address": "scripted-%s-worker",
                "protocol": "http",
                "path": "/",
                "discovery": {
                  "type": "static-env",
                  "hosts": [%s]
                }
                """.formatted(roleType.name().toLowerCase(), workerHosts(roleType));
    }

    private String workerHosts(RoleType roleType) {
        return IntStream.range(0, IntegrationTestFixtures.workerCount(roleType))
                .mapToObj(index -> "\"%s:%d\"".formatted(
                        IntegrationTestFixtures.WORKER_IP,
                        IntegrationTestFixtures.workerHttpPort(roleType, index)))
                .collect(Collectors.joining(", "));
    }

    private String endpointProperty(RoleType roleType) {
        return switch (roleType) {
            case PREFILL -> "prefill_endpoint";
            case DECODE -> "decode_endpoint";
            case VIT -> "vit_endpoint";
            case PDFUSION -> "pd_fusion_endpoint";
        };
    }

    private String kvcmConfig(
            int kvcmPort,
            long localStandbyTtlMs,
            long localStandbyMinimumTtlMs,
            long localStandbyMaximumEntries,
            int kvcmQueryFailureThreshold) {
        return """
                ,
                    "kvcm": {
                      "enabled": true,
                      "address": "scripted-kvcm",
                      "namespace": "flexlb-it",
                      "port": %d,
                      "request_timeout_ms": 500,
                      "leader_refresh_interval_ms": 50,
                      "heartbeat_failure_threshold": 3,
                      "query_failure_threshold": %d,
                      "max_query_retry_count": 0,
                      "recovery_success_threshold": 100,
                      "discovery": {
                        "type": "static-env",
                        "hosts": ["%s:%d"]
                      },
                      "local_standby": {
                        "auto_switch": true,
                        "block_size": 16,
                        "ttl_ms": %d,
                        "minimum_ttl_ms": %d,
                        "ttl_reduction_start_ratio": 0.8,
                        "maximum_entries": %d,
                        "capacity_multiplier": 1.0,
                        "async_queue_capacity": 100,
                        "hash_thread_count": 1,
                        "hash_queue_capacity": 100
                      }
                    }
                """.formatted(
                kvcmPort,
                kvcmQueryFailureThreshold,
                IntegrationTestFixtures.WORKER_IP,
                kvcmPort,
                localStandbyTtlMs,
                localStandbyMinimumTtlMs,
                localStandbyMaximumEntries);
    }
}
