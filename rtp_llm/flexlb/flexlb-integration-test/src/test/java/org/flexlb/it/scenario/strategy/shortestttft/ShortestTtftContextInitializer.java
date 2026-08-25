package org.flexlb.it.scenario.strategy.shortestttft;

import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/** Prepares the default direct-routing {@code SHORTEST_TTFT} test context. */
public final class ShortestTtftContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeShortestTtftContext(IntegrationTestFixtures.PDFUSION_TWO_WORKERS);
    }
}
