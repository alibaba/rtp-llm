package org.flexlb.it.scenario.strategy.cacheaffinity;

import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.configuration.EngineMode;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/** Prepares the default VLLM/KVCM {@code CACHE_AFFINITY_FIRST} application context. */
public final class CacheAffinityContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeCacheAffinityContext(IntegrationTestFixtures.PDFUSION_TWO_WORKERS, EngineMode.VLLM);
    }
}
