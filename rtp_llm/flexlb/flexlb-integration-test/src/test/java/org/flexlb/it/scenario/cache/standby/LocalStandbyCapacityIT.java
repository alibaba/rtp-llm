package org.flexlb.it.scenario.cache.standby;

import org.flexlb.Application;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.domain.CacheMatchStatus;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.cache.match.localstandby.LocalStandbyCacheManager;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.it.fixture.kvcm.KvcmIntegrationTestFixtures;
import org.flexlb.it.fixture.spring.CompletedWarmupConfiguration;
import org.flexlb.util.BlockCacheKeyCalculator;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
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
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Verifies the KVCM-to-Local-Standby capacity path using real routed-request metadata updates.
 *
 * <p>Local Standby deliberately does not evict unexpired mappings. At capacity it rejects a new
 * mapping, shortens retention to the configured minimum, removes expired metadata, and then
 * accepts a later routed-request update.
 */
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = LocalStandbyCapacityContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@Import(CompletedWarmupConfiguration.class)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
class LocalStandbyCapacityIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;
    private static final int[] FIRST_INPUT_IDS = IntStream.rangeClosed(1, 16).toArray();
    private static final int[] SECOND_INPUT_IDS = IntStream.rangeClosed(17, 32).toArray();

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @Autowired
    private CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;

    @Autowired
    private LocalStandbyCacheManager localStandbyCacheManager;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @BeforeEach
    void resetScriptedDependencies() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
        IntegrationTestFixtures.resetWorkers();
        KvcmIntegrationTestFixtures.setCacheResponse(KvcmIntegrationTestFixtures.CacheResponse.UNAVAILABLE);
        await("worker status and KVCM discovery").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(5)).until(() -> IntegrationTestFixtures.workerStatusCalls() > 0
                        && KvcmIntegrationTestFixtures.clusterInfoCalls() > 0);
    }

    @Test
    void should_admit_new_mapping_when_expired_standby_entry_is_evicted_at_capacity() {
        scheduleSuccessfully("standby-capacity-first-1", FIRST_INPUT_IDS);

        List<Long> firstBlockKeys = BlockCacheKeyCalculator.calculate(FIRST_INPUT_IDS, 16);
        List<Long> secondBlockKeys = BlockCacheKeyCalculator.calculate(SECOND_INPUT_IDS, 16);
        await("one routed Local Standby mapping after KVCM failover").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(2)).until(() ->
                        cacheMatchQueryOrchestrator.effectiveSource() == CacheMatchSource.LOCAL_STANDBY
                        && cacheMatchStatus().localStandbyEntries() == 1
                        && localStandbyCacheManager.findMatchingEngines(
                                firstBlockKeys,
                                SCHEDULING_ROLE,
                                "default").getOrDefault(
                                        IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, 0),
                                        0) > 0);

        waitForStandbyTtlWindow();
        scheduleSuccessfully("standby-capacity-second-rejected", SECOND_INPUT_IDS);
        await("expired Local Standby mapping eviction after capacity rejection").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(2)).until(() ->
                        cacheMatchStatus().localStandbyEntries() == 0
                        && !hasCachedBlocks(firstBlockKeys)
                        && !hasCachedBlocks(secondBlockKeys));

        scheduleSuccessfully("standby-capacity-second-admitted", SECOND_INPUT_IDS);
        await("new routed Local Standby mapping admission after cleanup").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(2))
                .until(() -> hasCachedBlocks(secondBlockKeys));

        CacheMatchStatus status = cacheMatchStatus();
        assertEquals(CacheMatchSource.LOCAL_STANDBY, status.effectiveSource());
        assertEquals(1, status.localStandbyMaximumEntries());
        assertEquals(1, status.localStandbyEntries());
        assertFalse(hasCachedBlocks(firstBlockKeys));
    }

    private void scheduleSuccessfully(String requestId, int[] inputIds) {
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", requestId,
                        "input_ids", inputIds,
                        "seq_len", inputIds.length,
                        "generate_timeout", 500))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true);
    }

    private CacheMatchStatus cacheMatchStatus() {
        CacheMatchStatus status = webTestClient.get()
                .uri("/flexlb/cache_match/status")
                .accept(MediaType.APPLICATION_JSON)
                .exchange()
                .expectStatus().isOk()
                .expectBody(CacheMatchStatus.class)
                .returnResult()
                .getResponseBody();
        if (status == null) {
            throw new IllegalStateException("Cache-match status response must not be null");
        }
        return status;
    }

    private void waitForStandbyTtlWindow() {
        long deadlineNanos = System.nanoTime() + Duration.ofMillis(600).toNanos();
        await("Local Standby TTL window").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(2)).until(() -> System.nanoTime() >= deadlineNanos);
    }

    private boolean hasCachedBlocks(List<Long> blockKeys) {
        return localStandbyCacheManager.findMatchingEngines(blockKeys, SCHEDULING_ROLE, "default")
                .values()
                .stream()
                .anyMatch(matchCount -> matchCount > 0);
    }
}
