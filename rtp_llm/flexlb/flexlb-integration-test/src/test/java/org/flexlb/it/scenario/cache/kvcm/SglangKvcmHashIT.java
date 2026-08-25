package org.flexlb.it.scenario.cache.kvcm;

import org.flexlb.Application;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.hash.SglangBlockHashStrategy;
import org.flexlb.cache.match.CacheMatchQueryOrchestrator;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.KvcmWorkerMetadataResolver;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.it.fixture.kvcm.KvcmIntegrationTestFixtures;
import org.flexlb.it.fixture.spring.CompletedWarmupConfiguration;
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
import java.util.Map;
import java.util.stream.IntStream;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Proves that the SGLang block-hash implementation produces the KVCM query keys for an
 * {@code input_ids} request.
 */
@ActiveProfiles("test")
@AutoConfigureWebTestClient
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = SglangKvcmContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@Import(CompletedWarmupConfiguration.class)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "logging.config=classpath:logback-it.xml")
class SglangKvcmHashIT {

    private static final int[] INPUT_IDS = IntStream.rangeClosed(101, 164).toArray();

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @Autowired
    private CacheMatchQueryOrchestrator cacheMatchQueryOrchestrator;

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
        IntegrationTestFixtures.setWorkerStatus(RoleType.PDFUSION, 1, true, 0, 0, 1.0);
        KvcmIntegrationTestFixtures.setCacheResponse(KvcmIntegrationTestFixtures.CacheResponse.EMPTY);
        await("SGLang KVCM worker metadata").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(5))
                .until(() -> workerMetadataResolver.resolveQueryType(RoleType.PDFUSION, "default") != null);
    }

    @Test
    void should_query_kvcm_with_sglang_hashes_when_request_provides_input_ids() {
        int callsBefore = KvcmIntegrationTestFixtures.cacheStateCalls();

        webTestClient.post()
                .uri("/rtp_llm/schedule")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(Map.of(
                        "request_id", "sglang-hash-to-kvcm",
                        "input_ids", INPUT_IDS,
                        "seq_len", 64,
                        "generate_timeout", 500))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.success").isEqualTo(true);

        await("SGLang KVCM cache-match request").pollDelay(Duration.ZERO).pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(2))
                .until(() -> KvcmIntegrationTestFixtures.cacheStateCalls() > callsBefore);
        assertEquals(new SglangBlockHashStrategy().calculate(INPUT_IDS, 16, 0),
                KvcmIntegrationTestFixtures.lastCacheBlockKeys());
        assertEquals(CacheMatchSource.KVCM, cacheMatchQueryOrchestrator.effectiveSource());
    }
}
