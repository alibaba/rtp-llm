package org.flexlb.service.grace.strategy;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.TrafficPolicyConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.service.grace.GracefulLifecycleReporter;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class QueryWarmerHookerTest {

    @Mock
    private GracefulLifecycleReporter lifecycleReporter;

    @Mock
    private ConfigService configService;

    private ModelMetaConfig modelMetaConfig;

    @BeforeEach
    void setUp() {
        modelMetaConfig = new ModelMetaConfig();
        QueryWarmerHooker.warmUpFinished = false;
        clearStatus();
        ModelMetaConfig.clearForTest();
        ModelMetaConfig.putServiceRoute("function.test-model", pdRoute());
    }

    @AfterEach
    void tearDown() {
        QueryWarmerHooker.warmUpFinished = false;
        clearStatus();
        ModelMetaConfig.clearForTest();
    }

    @Test
    void should_return_after_start_quickly_and_keep_health_offline_when_configured_role_is_not_ready() throws InterruptedException {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getPrefillStatusMap()
                .put("127.0.0.1:8080", freshAliveWorker("127.0.0.1", 8080));

        QueryWarmerHooker hooker = new QueryWarmerHooker(
                lifecycleReporter,
                modelMetaConfig,
                null,
                null,
                1000,
                1,
                1000,
                1
        );

        long startMs = System.currentTimeMillis();
        try {
            hooker.afterStartUp();
            assertTrue(System.currentTimeMillis() - startMs < 100);
            Thread.sleep(20);
            assertFalse(QueryWarmerHooker.warmUpFinished);
        } finally {
            hooker.stopWarmUp();
        }
    }

    @Test
    void should_mark_warmup_finished_after_all_configured_roles_are_ready() throws InterruptedException {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getPrefillStatusMap()
                .put("127.0.0.1:8080", freshAliveWorker("127.0.0.1", 8080));
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getDecodeStatusMap()
                .put("127.0.0.2:8081", freshAliveWorker("127.0.0.2", 8081));

        QueryWarmerHooker hooker = new QueryWarmerHooker(
                lifecycleReporter,
                modelMetaConfig,
                null,
                null,
                100,
                1,
                1000,
                2
        );

        try {
            hooker.afterStartUp();

            assertTrue(waitUntil(() -> QueryWarmerHooker.warmUpFinished));
            verify(lifecycleReporter).reportWarmerComplete(anyLong());
        } finally {
            hooker.stopWarmUp();
        }
    }

    @Test
    void should_mark_warmup_finished_when_configured_roles_use_different_routeable_groups() throws InterruptedException {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getPrefillStatusMap()
                .put("127.0.0.1:8080", freshAliveWorker("127.0.0.1", 8080, "group-a"));
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getDecodeStatusMap()
                .put("127.0.0.2:8081", freshAliveWorker("127.0.0.2", 8081, "group-b"));

        QueryWarmerHooker hooker = new QueryWarmerHooker(
                lifecycleReporter,
                modelMetaConfig,
                null,
                null,
                1000,
                1,
                1000,
                1
        );

        try {
            hooker.afterStartUp();
            assertTrue(waitUntil(() -> QueryWarmerHooker.warmUpFinished));
        } finally {
            hooker.stopWarmUp();
        }
    }

    @Test
    void should_not_block_health_on_partial_positive_traffic_group_observation() throws InterruptedException {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getPrefillStatusMap()
                .put("127.0.0.1:8080", freshAliveWorker("127.0.0.1", 8080, "old"));
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getDecodeStatusMap()
                .put("127.0.0.2:8081", freshAliveWorker("127.0.0.2", 8081, "old"));
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS
                .getPrefillStatusMap()
                .put("127.0.0.3:8080", freshAliveWorker("127.0.0.3", 8080, "new"));

        FlexlbConfig config = new FlexlbConfig();
        config.setTrafficPolicy(weightedTrafficPolicy("old", "new"));
        when(configService.loadBalanceConfig()).thenReturn(config);

        QueryWarmerHooker hooker = new QueryWarmerHooker(
                lifecycleReporter,
                modelMetaConfig,
                configService,
                null,
                1000,
                1,
                1000,
                1
        );

        try {
            hooker.afterStartUp();
            assertTrue(waitUntil(() -> QueryWarmerHooker.warmUpFinished));
        } finally {
            hooker.stopWarmUp();
        }
    }

    private static boolean waitUntil(BooleanSupplier predicate) throws InterruptedException {
        long deadlineMs = System.currentTimeMillis() + 1_000L;
        while (System.currentTimeMillis() < deadlineMs) {
            if (predicate.getAsBoolean()) {
                return true;
            }
            Thread.sleep(10);
        }
        return predicate.getAsBoolean();
    }

    private static WorkerStatus freshAliveWorker(String ip, int port) {
        return freshAliveWorker(ip, port, null);
    }

    private static WorkerStatus freshAliveWorker(String ip, int port, String group) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setIp(ip);
        workerStatus.setPort(port);
        workerStatus.setGroup(group);
        workerStatus.setAlive(true);
        workerStatus.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
        return workerStatus;
    }

    private static ServiceRoute pdRoute() {
        Endpoint prefill = new Endpoint();
        prefill.setAddress("prefill");
        Endpoint decode = new Endpoint();
        decode.setAddress("decode");

        GroupRoleEndPoint group = new GroupRoleEndPoint();
        group.setGroup("default");
        group.setPrefillEndpoint(prefill);
        group.setDecodeEndpoint(decode);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("function.test-model");
        route.setRoleEndpoints(List.of(group));
        return route;
    }

    private static TrafficPolicyConfig weightedTrafficPolicy(String... groups) {
        TrafficPolicyConfig trafficPolicy = new TrafficPolicyConfig();
        List<TrafficPolicyConfig.TrafficTargetGroup> targetGroups = new java.util.ArrayList<>();
        for (String groupName : groups) {
            TrafficPolicyConfig.TrafficTargetGroup group = new TrafficPolicyConfig.TrafficTargetGroup();
            group.setGroup(groupName);
            group.setWeight(50);
            targetGroups.add(group);
        }
        trafficPolicy.setDefaultTargetGroups(targetGroups);
        return trafficPolicy;
    }

    private static void clearStatus() {
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPrefillStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getDecodeStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getPdFusionStatusMap().clear();
        EngineWorkerStatus.MODEL_ROLE_WORKER_STATUS.getVitStatusMap().clear();
    }

    private interface BooleanSupplier {
        boolean getAsBoolean();
    }
}
