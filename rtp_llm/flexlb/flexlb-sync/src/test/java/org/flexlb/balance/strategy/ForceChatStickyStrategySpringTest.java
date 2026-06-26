package org.flexlb.balance.strategy;

import org.flexlb.balance.resource.ResourceMeasureFactory;
import org.flexlb.config.ModelMetaConfig;
import org.flexlb.sync.status.EngineWorkerStatus;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class ForceChatStickyStrategySpringTest {

    @Test
    void shouldCreateForceChatStickyStrategyBean() {
        try (AnnotationConfigApplicationContext context = new AnnotationConfigApplicationContext(
                TestConfig.class,
                ForceChatStickyStrategy.class)) {
            assertNotNull(context.getBean(ForceChatStickyStrategy.class));
        }
    }

    @Configuration
    static class TestConfig {
        @Bean
        EngineWorkerStatus engineWorkerStatus() {
            return new EngineWorkerStatus(new ModelMetaConfig());
        }

        @Bean
        ResourceMeasureFactory resourceMeasureFactory() {
            return Mockito.mock(ResourceMeasureFactory.class);
        }
    }
}
