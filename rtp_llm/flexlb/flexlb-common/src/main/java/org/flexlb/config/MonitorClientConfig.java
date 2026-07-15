
package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MonitorClientConfig {

    /**
     * Provides the fallback monitor. Property-selected providers expose a primary bean when active.
     */
    @Bean
    public FlexMonitor flexMonitor() {
        log.info("Creating fallback NoOpFlexMonitor");
        return NoOpFlexMonitor.getInstance();
    }
}
