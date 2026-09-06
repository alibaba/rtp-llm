package org.flexlb.engine.grpc.client;

import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.kvcm.grpc.QueryType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class KvcmWorkerMetadataResolverTest {

    @Test
    void resolvesConfiguredNamespaceAndDefaultQueryType() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        KvcmConfig config = new KvcmConfig();
        config.setNamespace("deployment-a");
        when(configuration.isKvcmEnabled()).thenReturn(true);
        when(configuration.getKvcmConfig()).thenReturn(config);

        KvcmWorkerMetadataResolver resolver = new KvcmWorkerMetadataResolver(
                configuration, mock(WorkerStatusProvider.class));

        assertTrue(resolver.usesConfiguredNamespace());
        assertEquals(
                "deployment-a_2192",
                resolver.resolveNamespace(RoleType.PREFILL, "default", 2192));
        assertEquals(
                QueryType.QT_PREFIX_MATCH,
                resolver.resolveQueryType(RoleType.PREFILL, "default"));
    }

    @Test
    void returnsNoNamespaceWhenKvcmIsDisabled() {
        CacheMatchConfiguration configuration = mock(CacheMatchConfiguration.class);
        when(configuration.isKvcmEnabled()).thenReturn(false);

        KvcmWorkerMetadataResolver resolver = new KvcmWorkerMetadataResolver(
                configuration, mock(WorkerStatusProvider.class));

        assertNull(resolver.resolveNamespace(RoleType.PREFILL, "default", 2192));
    }
}
