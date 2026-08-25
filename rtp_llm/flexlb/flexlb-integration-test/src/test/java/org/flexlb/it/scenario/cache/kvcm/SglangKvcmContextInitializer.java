package org.flexlb.it.scenario.cache.kvcm;

import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.configuration.EngineMode;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/** Prepares the SGLang hash-to-KVCM integration-test context. */
public final class SglangKvcmContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeCacheAffinityContext(IntegrationTestFixtures.PDFUSION_TWO_WORKERS, EngineMode.SGLANG);
    }
}
