package org.flexlb.it.scenario.strategy.cacheaffinity;

import org.flexlb.Application;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.KvcmWorkerMetadataResolver;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.it.fixture.kvcm.KvcmIntegrationTestFixtures;
import org.flexlb.it.fixture.spring.CompletedWarmupConfiguration;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.flexlb.util.BlockCacheKeyCalculator;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.test.web.reactive.server.WebTestClient;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.time.Duration;
import java.util.Map;
import java.util.stream.IntStream;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Covers {@code CACHE_AFFINITY_FIRST} with a real KVCM boundary and production failover state.
 *
 * <p>VLLM requests use {@code input_ids}; the test checks the resulting wire keys rather than
 * bypassing hash selection with supplied {@code block_cache_keys}. A cache leader is retained only
 * within its extra-work budget, outstanding-uncached-token guards, and minimum-hit-rate threshold.
 */
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = CacheAffinityContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@Import(CompletedWarmupConfiguration.class)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class CacheAffinityFirstIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;
    private static final int[] INPUT_IDS = IntStream.rangeClosed(1, 64).toArray();

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @Autowired
    private CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;

    @Autowired
    private ConfigService configService;

    @Autowired
    private KvcmWorkerMetadataResolver workerMetadataResolver;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @BeforeEach
    void resetScriptedDependencies() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
        IntegrationTestFixtures.resetWorkers();
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 1, true, 0, 0, 1.0);
        KvcmIntegrationTestFixtures.setCacheResponse(KvcmIntegrationTestFixtures.CacheResponse.EMPTY);
        KvcmIntegrationTestFixtures.setLocalMatchBlocks(-1);
        configureCacheAffinityGuards(1_000_000, null, 0, 0);
        await("worker metadata and KVCM leader discovery").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5)).until(() -> workerAlive(0)
                        && workerAlive(1)
                        && workerMetadataResolver.resolveQueryType(RoleType.PDFUSION, "default") != null
                        && KvcmIntegrationTestFixtures.clusterInfoCalls() > 0);
    }

    @Test
    @Order(1)
    void should_select_cache_leader_when_extra_work_is_within_budget() {
        publishLongBucketStatusForCacheLeader();
        KvcmIntegrationTestFixtures.setCacheResponse(
                KvcmIntegrationTestFixtures.CacheResponse.CONFIGURED_WORKER_MATCH);
        int callsBefore = KvcmIntegrationTestFixtures.cacheStateCalls();

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request("cache-affinity-leader"))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].http_port")
                .isEqualTo(IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, 1));

        assertEquals(CacheMatchSource.KVCM, cacheMatchQueryOrchestrator.effectiveSource());
        await("KVCM cache-match request").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(2))
                .until(() -> KvcmIntegrationTestFixtures.cacheStateCalls() > callsBefore);
        assertEquals(BlockCacheKeyCalculator.calculate(INPUT_IDS, 16),
                KvcmIntegrationTestFixtures.lastCacheBlockKeys());
    }

    @Test
    @Order(2)
    void should_select_shortest_ttft_worker_when_cache_leader_exceeds_extra_work_budget() {
        configureCacheAffinityGuards(0, null, 0, 0);
        publishLongBucketStatusForCacheLeader();
        KvcmIntegrationTestFixtures.setCacheResponse(
                KvcmIntegrationTestFixtures.CacheResponse.CONFIGURED_WORKER_MATCH);

        scheduleToWorker("cache-affinity-extra-work-exceeded", 0);

        assertEquals(CacheMatchSource.KVCM, cacheMatchQueryOrchestrator.effectiveSource());
    }

    @Test
    @Order(3)
    void should_select_shortest_ttft_worker_when_neutral_outstanding_uncached_tokens_threshold_is_exceeded() {
        configureCacheAffinityGuards(1_000_000, 100L, 0, 0);
        publishLongBucketStatusForCacheLeader();
        KvcmIntegrationTestFixtures.setCacheResponse(
                KvcmIntegrationTestFixtures.CacheResponse.CONFIGURED_WORKER_MATCH);

        scheduleToWorker("cache-affinity-neutral-outstanding-guard", 0);

        assertEquals(CacheMatchSource.KVCM, cacheMatchQueryOrchestrator.effectiveSource());
    }

    @Test
    @Order(4)
    void should_select_shortest_ttft_worker_when_legacy_outstanding_uncached_tokens_threshold_is_exceeded() {
        configureCacheAffinityGuards(1_000_000, null, 100, 0);
        publishLongBucketStatusForCacheLeader();
        KvcmIntegrationTestFixtures.setCacheResponse(
                KvcmIntegrationTestFixtures.CacheResponse.CONFIGURED_WORKER_MATCH);

        scheduleToWorker("cache-affinity-legacy-outstanding-guard", 0);

        assertEquals(CacheMatchSource.KVCM, cacheMatchQueryOrchestrator.effectiveSource());
    }

    @Test
    @Order(5)
    void should_select_shortest_ttft_worker_when_cache_leader_hit_rate_is_below_minimum() {
        configureCacheAffinityGuards(1_000_000, null, 0, 50);
        publishLongBucketStatusForCacheLeader();
        KvcmIntegrationTestFixtures.setCacheResponse(
                KvcmIntegrationTestFixtures.CacheResponse.CONFIGURED_WORKER_MATCH);
        KvcmIntegrationTestFixtures.setLocalMatchBlocks(1);

        scheduleToWorker("cache-affinity-low-hit-rate", 0);

        assertEquals(CacheMatchSource.KVCM, cacheMatchQueryOrchestrator.effectiveSource());
    }

    @Test
    @Order(6)
    void should_keep_kvcm_active_when_cache_query_returns_valid_empty_result() {
        publishLongBucketStatusForCacheLeader();
        int callsBefore = KvcmIntegrationTestFixtures.cacheStateCalls();

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request("kvcm-empty-result"))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].http_port")
                .isEqualTo(IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, 0));

        await("empty KVCM cache-match response").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(2))
                .until(() -> KvcmIntegrationTestFixtures.cacheStateCalls() > callsBefore);
        assertEquals(CacheMatchSource.KVCM, cacheMatchQueryOrchestrator.effectiveSource());
    }

    @Test
    @Order(7)
    void should_complete_scheduling_through_local_standby_when_kvcm_query_failure_threshold_is_reached() {
        KvcmIntegrationTestFixtures.setCacheResponse(KvcmIntegrationTestFixtures.CacheResponse.UNAVAILABLE);

        scheduleSuccessfully("kvcm-query-failure-1");
        scheduleSuccessfully("kvcm-query-failure-2");
        await("automatic Local Standby activation after KVCM query failures").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(2))
                .until(() -> cacheMatchQueryOrchestrator.effectiveSource() == CacheMatchSource.LOCAL_STANDBY);
        int callsAfterFailure = KvcmIntegrationTestFixtures.cacheStateCalls();

        scheduleSuccessfully("local-standby-active");

        assertEquals(callsAfterFailure, KvcmIntegrationTestFixtures.cacheStateCalls());
        assertEquals(CacheMatchSource.LOCAL_STANDBY, cacheMatchQueryOrchestrator.effectiveSource());
    }

    private void scheduleSuccessfully(String requestId) {
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request(requestId))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true);
    }

    private void scheduleToWorker(String requestId, int workerIndex) {
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request(requestId))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].http_port")
                .isEqualTo(IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, workerIndex));
    }

    @SuppressWarnings("deprecation")
    private void configureCacheAffinityGuards(
            long maxExtraWorkTokens,
            Long outstandingUncachedTokensThreshold,
            long legacyOutstandingUncachedTokensThreshold,
            double minimumHitRate) {
        var config = configService.loadBalanceConfig();
        config.setCacheAffinityFirstMaxExtraWorkTokens(maxExtraWorkTokens);
        config.setOutstandingUncachedTokensThreshold(outstandingUncachedTokensThreshold);
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(
                legacyOutstandingUncachedTokensThreshold);
        config.setCacheAffinityFirstMinHitRate(minimumHitRate);
    }

    private void publishLongBucketStatusForCacheLeader() {
        IntegrationTestFixtures.setWorkerLongPrefillStatus(SCHEDULING_ROLE, 1);
        await("cache leader long-bucket outstanding work").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                .until(() -> workerOutstandingUncachedTokens(1) > 100);
    }

    private Map<String, Object> request(String requestId) {
        return Map.of(
                "request_id", requestId,
                "input_ids", INPUT_IDS,
                "seq_len", 64,
                "generate_timeout", 500);
    }

    private boolean workerAlive(int workerIndex) {
        var worker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .get(IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, workerIndex));
        return worker != null && worker.isAlive();
    }

    private long workerOutstandingUncachedTokens(int workerIndex) {
        var worker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .get(IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, workerIndex));
        return worker == null ? 0 : worker.getOutstandingUncachedTokens();
    }
}
