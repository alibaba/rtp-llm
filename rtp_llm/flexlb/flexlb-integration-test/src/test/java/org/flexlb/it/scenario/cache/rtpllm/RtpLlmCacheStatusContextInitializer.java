package org.flexlb.it.scenario.cache.rtpllm;

import org.flexlb.it.configuration.AbstractIntegrationTestContextInitializer;
import org.flexlb.it.configuration.EngineMode;
import org.flexlb.it.fixture.engine.IntegrationTestFixtures;
import org.springframework.context.ApplicationContextInitializer;
import org.springframework.context.ConfigurableApplicationContext;

/** Prepares the RTP-LLM local cache-status context without a KVCM boundary. */
public final class RtpLlmCacheStatusContextInitializer
        extends AbstractIntegrationTestContextInitializer
        implements ApplicationContextInitializer<ConfigurableApplicationContext> {

    @Override
    public void initialize(ConfigurableApplicationContext applicationContext) {
        initializeCacheAffinityContext(IntegrationTestFixtures.PDFUSION_TWO_WORKERS, EngineMode.RTP_LLM);
    }
}
