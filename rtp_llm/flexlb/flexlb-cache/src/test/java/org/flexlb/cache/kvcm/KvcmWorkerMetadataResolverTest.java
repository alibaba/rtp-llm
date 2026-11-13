package org.flexlb.cache.kvcm;

import org.flexlb.cache.kvcm.protocol.QueryType;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.cache.CacheMatchConfiguration;
import org.flexlb.config.cache.KvcmCacheMatchingConfig;
import org.flexlb.config.cache.KvcmConfig;
import org.flexlb.config.service.GroupRoleEndPoint;
import org.flexlb.config.service.ModelMetaConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.config.service.ServiceRoute;
import org.flexlb.util.discovery.DiscoveryEndpoint;
import org.flexlb.worker.cache.KvCacheGroupMode;
import org.flexlb.worker.status.WorkerStatus;
import org.flexlb.worker.status.WorkerStatusProvider;
import org.flexlb.worker.status.WorkerStatusResponse;
import org.junit.jupiter.api.Test;

import java.util.Collection;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Checks resolution of worker metadata required by KVCM queries.
 */
class KvcmWorkerMetadataResolverTest {
    @Test
    void resolvesNamespaceAndQueryTypeFromOneWorkerStatusRefresh() {
        WorkerStatus workerStatus = workerStatus("prefill-deployment", KvCacheGroupMode.WITH_MAMBA);
        KvcmWorkerMetadataResolver resolver =
                new KvcmWorkerMetadataResolver(
                        configuration(null), provider(List.of(workerStatus)));

        resolver.refreshNamespacesAndQueryTypes();

        assertEquals(
                "prefill-deployment_2192",
                resolver.resolveNamespace(RoleType.PREFILL, "default", 2192));
        assertEquals(
                QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                resolver.resolveQueryType(RoleType.PREFILL, "default"));
        assertEquals(
                QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                resolver.resolveQueryType(RoleType.PREFILL, null));
    }

    @Test
    void configuredNamespaceTakesPriorityWhileQueryTypeStillUsesWorkerStatus() {
        WorkerStatus workerStatus =
                workerStatus("ignored-deployment", KvCacheGroupMode.FULL_ATTENTION_ONLY);
        KvcmWorkerMetadataResolver resolver =
                new KvcmWorkerMetadataResolver(
                        configuration("configured-namespace"), provider(List.of(workerStatus)));

        resolver.refreshNamespacesAndQueryTypes();

        assertEquals(
                "configured-namespace_1024",
                resolver.resolveNamespace(RoleType.PREFILL, "default", 1024));
        assertEquals(
                QueryType.QT_PREFIX_MATCH, resolver.resolveQueryType(RoleType.PREFILL, "default"));
    }

    @Test
    void unspecifiedModeDoesNotOverwriteLastTrustedQueryType() {
        WorkerStatus workerStatus = workerStatus("prefill-deployment", KvCacheGroupMode.WITH_MAMBA);
        KvcmWorkerMetadataResolver resolver =
                new KvcmWorkerMetadataResolver(
                        configuration(null), provider(List.of(workerStatus)));
        resolver.refreshNamespacesAndQueryTypes();

        publishMode(workerStatus, KvCacheGroupMode.UNSPECIFIED, 2L);
        resolver.refreshNamespacesAndQueryTypes();

        assertEquals(
                QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                resolver.resolveQueryType(RoleType.PREFILL, "default"));
    }

    private WorkerStatus workerStatus(String deploymentName, KvCacheGroupMode mode) {
        WorkerStatus workerStatus =
                WorkerStatus.createDiscovered(
                        RoleType.PREFILL,
                        "default",
                        "127.0.0.1",
                        8080,
                        8081,
                        "test",
                        deploymentName);
        publishMode(workerStatus, mode, 1L);
        return workerStatus;
    }

    private void publishMode(WorkerStatus workerStatus, KvCacheGroupMode mode, long statusVersion) {
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRole(RoleType.PREFILL);
        response.setAlive(true);
        response.setStatusVersion(statusVersion);
        response.setLatestFinishedVersion(0L);
        response.setRunningTaskInfo(Map.of());
        response.setFinishedTaskInfo(Map.of());
        response.setKvCacheGroupMode(mode);

        workerStatus.lock.lock();
        try {
            WorkerStatus.PreparedStatus prepared =
                    workerStatus.prepareStatus(
                            org.flexlb.worker.status.WorkerStatusTestSupport.freezeStatusResponse(
                                    workerStatus, response));
            workerStatus.publishPreparedStatus(prepared);
            workerStatus.recordSuccessfulPoll(true);
        } finally {
            workerStatus.lock.unlock();
        }
    }

    private WorkerStatusProvider provider(Collection<WorkerStatus> workerStatuses) {
        return new WorkerStatusProvider() {
            @Override
            public Collection<WorkerStatus> getWorkerStatuses(RoleType roleType, String group) {
                return workerStatuses;
            }
        };
    }

    private CacheMatchConfiguration configuration(String namespace) {
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        flexlbConfig.setCacheMatching(new KvcmCacheMatchingConfig());
        return new CacheMatchConfiguration(modelMetaConfig(namespace), flexlbConfig);
    }

    private ModelMetaConfig modelMetaConfig(String namespace) {
        DiscoveryEndpoint endpoint = new DiscoveryEndpoint();
        endpoint.setAddress("v-prefill");

        GroupRoleEndPoint group = new GroupRoleEndPoint();
        group.setGroup("default");
        group.setPrefillEndpoint(endpoint);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setNamespace(namespace);

        ServiceRoute route = new ServiceRoute();
        route.setServiceId("test-service");
        route.setKvcm(kvcm);
        route.setRoleEndpoints(List.of(group));

        ModelMetaConfig modelMetaConfig = new ModelMetaConfig();
        modelMetaConfig.putServiceRoute(route.getServiceId(), route);
        return modelMetaConfig;
    }
}
