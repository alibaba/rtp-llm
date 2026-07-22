package org.flexlb.engine.grpc.client;

import org.flexlb.config.ModelMetaConfig;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.Endpoint;
import org.flexlb.dao.route.GroupRoleEndPoint;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.kvcm.grpc.QueryType;
import org.junit.jupiter.api.Test;

import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class KvcmWorkerMetadataResolverTest {

    @Test
    void resolvesNamespaceAndQueryTypeFromOneWorkerStatusRefresh() {
        WorkerStatus workerStatus = workerStatus(
                "prefill-deployment", KvCacheGroupMode.WITH_MAMBA);
        KvcmWorkerMetadataResolver resolver = new KvcmWorkerMetadataResolver(
                modelMetaConfig(null), provider(List.of(workerStatus)));

        resolver.refresh();

        assertEquals("prefill-deployment_2192",
                resolver.resolveNamespace(RoleType.PREFILL, "default", 2192));
        assertEquals(QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                resolver.resolveQueryType(RoleType.PREFILL, "default"));
        assertEquals(QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                resolver.resolveQueryType(RoleType.PREFILL, null));
    }

    @Test
    void configuredNamespaceTakesPriorityWhileQueryTypeStillUsesWorkerStatus() {
        WorkerStatus workerStatus = workerStatus(
                "ignored-deployment", KvCacheGroupMode.FULL_ATTENTION_ONLY);
        KvcmWorkerMetadataResolver resolver = new KvcmWorkerMetadataResolver(
                modelMetaConfig("configured-namespace"), provider(List.of(workerStatus)));

        resolver.refresh();

        assertEquals("configured-namespace_1024",
                resolver.resolveNamespace(RoleType.PREFILL, "default", 1024));
        assertEquals(QueryType.QT_PREFIX_MATCH,
                resolver.resolveQueryType(RoleType.PREFILL, "default"));
    }

    @Test
    void unspecifiedModeDoesNotOverwriteLastTrustedQueryType() {
        WorkerStatus workerStatus = workerStatus(
                "prefill-deployment", KvCacheGroupMode.WITH_MAMBA);
        KvcmWorkerMetadataResolver resolver = new KvcmWorkerMetadataResolver(
                modelMetaConfig(null), provider(List.of(workerStatus)));
        resolver.refresh();

        workerStatus.setKvCacheGroupMode(KvCacheGroupMode.UNSPECIFIED);
        resolver.refresh();

        assertEquals(QueryType.QT_PREFIX_MATCH_WITH_MAMBA,
                resolver.resolveQueryType(RoleType.PREFILL, "default"));
    }

    private WorkerStatus workerStatus(String deploymentName, KvCacheGroupMode mode) {
        WorkerStatus workerStatus = new WorkerStatus();
        workerStatus.setDeploymentName(deploymentName);
        workerStatus.setKvCacheGroupMode(mode);
        return workerStatus;
    }

    private WorkerStatusProvider provider(Collection<WorkerStatus> workerStatuses) {
        return new WorkerStatusProvider() {
            @Override
            public Collection<WorkerStatus> getWorkerStatuses(RoleType roleType, String group) {
                return workerStatuses;
            }
        };
    }

    private ModelMetaConfig modelMetaConfig(String namespace) {
        Endpoint endpoint = new Endpoint();
        endpoint.setAddress("v-prefill");

        GroupRoleEndPoint group = new GroupRoleEndPoint();
        group.setGroup("default");
        group.setPrefillEndpoint(endpoint);

        KvcmConfig kvcm = new KvcmConfig();
        kvcm.setEnabled(true);
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
