package org.flexlb.cache;

import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.KvcmCacheMatchingConfig;
import org.flexlb.config.ModelMetaConfig;

import java.util.function.Consumer;

/** Test builders that keep topology and FlexLB behavior configuration separate. */
public final class CacheMatchTestConfigurations {

    private CacheMatchTestConfigurations() {
    }

    public static CacheMatchConfiguration localSync(ModelMetaConfig modelMetaConfig) {
        return new CacheMatchConfiguration(modelMetaConfig, new FlexlbConfig());
    }

    public static CacheMatchConfiguration kvcm(ModelMetaConfig modelMetaConfig) {
        return kvcm(modelMetaConfig, ignored -> { });
    }

    public static CacheMatchConfiguration kvcm(
            ModelMetaConfig modelMetaConfig,
            Consumer<KvcmCacheMatchingConfig> customizer) {
        KvcmCacheMatchingConfig kvcm = new KvcmCacheMatchingConfig();
        customizer.accept(kvcm);
        FlexlbConfig flexlbConfig = new FlexlbConfig();
        flexlbConfig.setCacheMatching(kvcm);
        return new CacheMatchConfiguration(modelMetaConfig, flexlbConfig);
    }
}
