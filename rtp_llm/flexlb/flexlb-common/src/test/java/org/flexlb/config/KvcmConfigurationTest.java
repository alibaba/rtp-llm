package org.flexlb.config;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class KvcmConfigurationTest {
    private static final String BASE = "\"requestLifecycle\":{\"request\":{\"timeoutMs\":5000}}";
    @Test void defaultAndExplicitOffUseLocalSync() {
        assertInstanceOf(LocalSyncCacheMatchingConfig.class, ConfigService.parse("{" + BASE + "}").getCacheMatching());
        assertInstanceOf(LocalSyncCacheMatchingConfig.class,
                ConfigService.parse("{" + BASE + ",\"cacheMatching\":{\"type\":\"LOCAL_SYNC\"}}").getCacheMatching());
    }
    @Test void onUsesMuModeNameWithoutEnablingRemotePayloadIo() {
        assertInstanceOf(KvcmCacheMatchingConfig.class,
                ConfigService.parse("{" + BASE + ",\"cacheMatching\":{\"type\":\"KVCM\",\"medium\":[\"hbm\"],\"globalKvsHostCount\":0,\"enableP2p\":false}}").getCacheMatching());
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("{" + BASE
                + ",\"cacheMatching\":{\"type\":\"KVCM\",\"requestTimeoutMs\":0}}"));
        assertThrows(ConfigValidationException.class, () -> ConfigService.parse("{" + BASE
                + ",\"cacheMatching\":{\"type\":\"LOCAL_SYNC\",\"medium\":[\"hbm\"]}}"));
    }
}
