
package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MonitorClientConfig {

    /**
     * Create FlexMonitor instance
     * Open-source builds use NoOpFlexMonitor. Internal builds provide their own implementation and
     * use FLEXLB_MONITOR_ENABLED to select KMonitor or NoOpFlexMonitor.
     */
    @Bean
    @ConditionalOnMissingBean(FlexMonitor.class)
    @ConditionalOnMissingClass("com.taobao.kmonitor.KMonitor")
    public FlexMonitor flexMonitor() {
        log.info("Creating default NoOpFlexMonitor - monitoring disabled");
        return NoOpFlexMonitor.getInstance();
    }
}
