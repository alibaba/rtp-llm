package org.flexlb.it.scenario.cache.rtpllm;

import org.flexlb.Application;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
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
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Covers the RTP-LLM caller-provided {@code block_cache_keys} contract.
 *
 * <p>The assertion first waits for the real {@code GetCacheStatus} path to populate local cache
 * state, then proves that routing uses {@code LOCAL_SYNC}; RTP-LLM does not participate in a
 * synthetic FlexLB hash-strategy test.
 */
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = RtpLlmCacheStatusContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
class RtpLlmCacheStatusIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;
    private static final long[] PROVIDED_BLOCK_CACHE_KEYS = {91L, 92L, 93L, 94L};

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @Autowired
    private CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;

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
        IntegrationTestFixtures.setWorkerCacheKeys(SCHEDULING_ROLE, 1, PROVIDED_BLOCK_CACHE_KEYS);
        IntegrationTestFixtures.setWorkerLongPrefillStatus(SCHEDULING_ROLE, 1);
        await("RTP-LLM worker cache-status synchronization").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                .until(this::cacheLeaderStatusContainsProvidedKeys);
    }

    @Test
    void should_route_using_local_cache_status_when_request_provides_rtp_block_cache_keys() {
        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "rtp-llm-cache-status",
                        "block_cache_keys", PROVIDED_BLOCK_CACHE_KEYS,
                        "block_size", 16,
                        "seq_len", 64,
                        "generate_timeout", 500))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true)
                .jsonPath("$.server_status[0].http_port")
                .isEqualTo(IntegrationTestFixtures.workerHttpPort(SCHEDULING_ROLE, 1));

        assertEquals(CacheMatchSource.LOCAL_SYNC, cacheMatchQueryOrchestrator.effectiveSource());
        assertTrue(IntegrationTestFixtures.cacheStatusCalls(SCHEDULING_ROLE) > 0);
    }

    private boolean cacheLeaderStatusContainsProvidedKeys() {
        WorkerStatus workerStatus = EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getRoleStatusMap(SCHEDULING_ROLE)
                .get(IntegrationTestFixtures.workerIpPort(SCHEDULING_ROLE, 1));
        return workerStatus != null
                && workerStatus.getCacheStatus() != null
                && workerStatus.getCacheStatus().getCachedKeys() != null
                && workerStatus.getCacheStatus().getCachedKeys().containsAll(Set.of(91L, 92L, 93L, 94L));
    }
}
