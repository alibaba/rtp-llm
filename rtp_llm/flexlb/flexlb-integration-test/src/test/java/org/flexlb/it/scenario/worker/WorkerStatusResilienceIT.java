package org.flexlb.it.scenario.worker;

import io.grpc.Status;
import org.flexlb.Application;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.it.scenario.strategy.shortestttft.ShortestTtftContextInitializer;
import org.junit.jupiter.api.AfterEach;
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

import static org.awaitility.Awaitility.await;

/**
 * Confirms that temporary worker-status RPC failures do not prevent healthy workers from serving
 * short requests.
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
class WorkerStatusResilienceIT {

    private static final RoleType SCHEDULING_ROLE = RoleType.PDFUSION;

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @Autowired
    private WebTestClient webTestClient;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @BeforeEach
    void resetScriptedWorkers() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
        IntegrationTestFixtures.resetWorkers();
        IntegrationTestFixtures.setWorkerStatus(SCHEDULING_ROLE, 1, true, 0, 0, 1.0);
    }

    @AfterEach
    void clearScriptedFailure() {
        IntegrationTestFixtures.clearWorkerStatusFailures();
    }

    @Test
    void should_schedule_short_bucket_requests_when_one_worker_status_rpc_is_unavailable() {
        await("initial worker-status synchronization").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(5))
                .until(() -> IntegrationTestFixtures.workerStatusCalls() > 0);
        int callsBeforeFailure = IntegrationTestFixtures.workerStatusCalls();
        IntegrationTestFixtures.failWorkerStatus(SCHEDULING_ROLE, 0, Status.UNAVAILABLE
                .withDescription("scripted transient worker-status outage")
                .asRuntimeException());
        await("repeated worker-status polling during the outage").pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10)).atMost(Duration.ofSeconds(2))
                .until(() -> IntegrationTestFixtures.workerStatusCalls() >= callsBeforeFailure + 2);

        for (int requestIndex = 0; requestIndex < 20; requestIndex++) {
            webTestClient.post()
                    .uri("/rtp_llm/schedule")
                    .contentType(MediaType.APPLICATION_JSON)
                    .bodyValue(request("short-bucket-status-outage-" + requestIndex))
                    .exchange()
                    .expectStatus().isOk()
                    .expectBody()
                    .jsonPath("$.success").isEqualTo(true);
        }
    }

    private Map<String, Object> request(String requestId) {
        return Map.of(
                "request_id", requestId,
                "block_cache_keys", new long[]{11L, 22L},
                "block_size", 16,
                "seq_len", 32,
                "generate_timeout", 500);
    }
}
