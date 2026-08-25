package org.flexlb.it.scenario.worker;

import org.flexlb.Application;
import org.flexlb.dao.route.RoleType;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.context.ContextConfiguration;
import uk.org.webcompere.systemstubs.environment.EnvironmentVariables;
import uk.org.webcompere.systemstubs.jupiter.SystemStub;
import uk.org.webcompere.systemstubs.jupiter.SystemStubsExtension;

import java.time.Duration;
import java.util.Map;

import static org.awaitility.Awaitility.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Verifies that a context can declare independent worker counts for each {@link RoleType}. */
@ActiveProfiles("test")
@ExtendWith(SystemStubsExtension.class)
@ContextConfiguration(initializers = MultiRoleContextInitializer.class)
@DirtiesContext(classMode = DirtiesContext.ClassMode.AFTER_CLASS)
@SpringBootTest(
        classes = Application.class,
        webEnvironment = SpringBootTest.WebEnvironment.NONE,
        properties = "logging.config=classpath:logback-it.xml")
class RoleAwareWorkerTopologyIT {

    private static final Map<RoleType, Integer> EXPECTED_WORKER_COUNTS = Map.of(
            RoleType.PREFILL, 2,
            RoleType.DECODE, 3,
            RoleType.PDFUSION, 1);

    @SystemStub
    private static EnvironmentVariables environmentVariables;

    @BeforeAll
    static void configureEnvironment() {
        environmentVariables.set("HIPPO_ROLE", "flexlb-it");
        environmentVariables.remove("FLEXLB_NACOS_SERVER_ADDR");
    }

    @Test
    void should_synchronize_role_specific_worker_counts_when_topology_is_declared() {
        await().alias("role-aware worker-status synchronization")
                .pollDelay(Duration.ZERO)
                .pollInterval(Duration.ofMillis(10))
                .atMost(Duration.ofSeconds(5))
                .until(() -> EXPECTED_WORKER_COUNTS.entrySet().stream().allMatch(entry ->
                        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                                .getRoleStatusMap(entry.getKey())
                                .size() == entry.getValue()));

        EXPECTED_WORKER_COUNTS.forEach((roleType, expectedCount) -> {
            assertEquals(expectedCount, IntegrationTestFixtures.workerCount(roleType));
            assertTrue(IntegrationTestFixtures.workerStatusCalls(roleType) >= expectedCount);
            assertTrue(EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                    .getRoleStatusMap(roleType)
                    .values()
                    .stream()
                    .allMatch(worker -> roleType.getCode().equals(worker.getRole())));
        });
    }
}
