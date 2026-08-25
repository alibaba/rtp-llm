package org.flexlb.it.scenario.strategy.shortestttft;

import org.flexlb.Application;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
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
import java.util.Set;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies the {@code SHORTEST_TTFT} cache-preference boundary controlled by similarity ratio.
 *
 * <p>The lower-TTFT worker costs 75 tokens without a cache hit. The other worker costs 100 tokens
 * from its reported running work but has a complete 75-token local cache hit. Their 25-token gap
 * is outside the 0.2 threshold and inside the 0.5 and 0.8 thresholds.
 */
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = ShortestTtftContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
class ShortestTtftSimilarityRatioIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;
    private static final long[] FULLY_CACHED_BLOCK_KEYS = {91L, 92L, 93L, 94L, 95L};
    private static final int REQUEST_TOKENS = 75;
    private static final int CACHED_WORKER_QUEUE_TOKENS = 100;

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @Autowired
    private ConfigService configService;

    @Autowired
    private CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @BeforeEach
    void prepareSimilarityBoundary() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
        IntegrationTestFixtures.resetWorkers();
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 0, true, 0, 0, 1.0);
        IntegrationTestFixtures.setWorkerCacheKeys(SCHEDULING_ROLE, 1, FULLY_CACHED_BLOCK_KEYS);
        IntegrationTestFixtures.setWorkerRunningPrefillStatus(
                SCHEDULING_ROLE,
                1,
                CACHED_WORKER_QUEUE_TOKENS);
        configService.loadBalanceConfig().setShortestTtftSimilarityThresholdRatio(0.2);
        await("similarity-ratio worker status and local cache synchronization").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                .until(this::similarityBoundaryIsSynchronized);
    }

    @ParameterizedTest(name = "should_select_worker_{1}_when_similarity_ratio_is_{0}")
    @CsvSource({"0.2, 0", "0.5, 1", "0.8, 1"})
    void should_select_expected_worker_when_similarity_ratio_is_applied(
            double ratio, int expectedWorkerIndex) {
        scheduleWithSimilarityRatio(ratio, expectedWorkerIndex);
    }

    private void scheduleWithSimilarityRatio(double ratio, int expectedWorkerIndex) {
        configService.loadBalanceConfig().setShortestTtftSimilarityThresholdRatio(ratio);

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "similarity-ratio-" + ratio,
                        "block_cache_keys", FULLY_CACHED_BLOCK_KEYS,
                        "block_size", 16,
                        "seq_len", REQUEST_TOKENS,
                        "generate_timeout", 500))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].http_port")
                .isEqualTo(IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, expectedWorkerIndex));

        assertEquals(CacheMatchSource.LOCAL_SYNC, cacheMatchQueryOrchestrator.effectiveSource());
    }

    private boolean similarityBoundaryIsSynchronized() {
        WorkerStatus cachedWorker = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .get(IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, 1));
        return cachedWorker != null
                && cachedWorker.isAlive()
                && cachedWorker.getRunningQueueTime().get() == CACHED_WORKER_QUEUE_TOKENS
                && cachedWorker.getCacheStatus() != null
                && cachedWorker.getCacheStatus().getCachedKeys() != null
                && cachedWorker.getCacheStatus().getCachedKeys().containsAll(Set.of(91L, 92L, 93L, 94L, 95L));
    }
}
