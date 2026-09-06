package org.flexlb.engine.grpc.client;

import org.apache.commons.lang3.StringUtils;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerStatusProvider;
import org.flexlb.dao.route.KvcmConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.kvcm.grpc.QueryType;
import org.springframework.stereotype.Component;

/**
 * Resolves the explicitly configured KVCM namespace for the foundation stage.
 * Worker-advertised namespace and attention metadata are enabled when the
 * worker lifecycle metadata lands in the following feature commit.
 */
@Component
public class KvcmWorkerMetadataResolver {

    private static final QueryType DEFAULT_QUERY_TYPE = QueryType.QT_PREFIX_MATCH;

    private final String configuredNamespace;

    public KvcmWorkerMetadataResolver(
            CacheMatchConfiguration configuration,
            WorkerStatusProvider workerStatusProvider) {
        KvcmConfig config = configuration.getKvcmConfig();
        this.configuredNamespace = configuration.isKvcmEnabled()
                ? StringUtils.trimToNull(config.getNamespace())
                : null;
    }

    public String resolveNamespace(RoleType roleType, String group, long blockSize) {
        return configuredNamespace == null
                ? null
                : configuredNamespace + "_" + blockSize;
    }

    public QueryType resolveQueryType(RoleType roleType, String group) {
        return DEFAULT_QUERY_TYPE;
    }

    public boolean usesConfiguredNamespace() {
        return configuredNamespace != null;
    }

    public void refreshNamespacesAndQueryTypes() {
        // Worker metadata is introduced by the next lifecycle feature commit.
    }
}
