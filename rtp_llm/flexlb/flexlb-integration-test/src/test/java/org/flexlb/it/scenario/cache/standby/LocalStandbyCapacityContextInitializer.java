package org.flexlb.it.scenario.cache.standby;

import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.configuration.EngineMode;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/**
 * Prepares a KVCM context whose one-entry Local Standby index expires mappings quickly under
 * capacity pressure.
 */
public final class LocalStandbyCapacityContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeCacheAffinityContext(
                IntegrationTestFixtures.PDFUSION_TWO_WORKERS,
                EngineMode.VLLM,
                500,
                500,
                1,
                1);
    }
}
