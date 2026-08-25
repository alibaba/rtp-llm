package org.flexlb.it.fixture.spring;

import org.flexlb.listener.ApplicationWarmupState;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Primary;

/**
 * Test-only warmup state used where a routing scenario must start after application warmup.
 */
@TestConfiguration(proxyBeanMethods = false)
public class CompletedWarmupConfiguration {

    @Bean
    @Primary
    ApplicationWarmupState completedApplicationWarmupState() {
        return () -> true;
    }
}
