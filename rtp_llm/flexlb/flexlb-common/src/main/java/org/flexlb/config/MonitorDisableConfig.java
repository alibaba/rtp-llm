package org.flexlb.config;

import io.micrometer.core.instrument.config.MeterFilter;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.Primary;

@Configuration
public class MonitorDisableConfig {
    @Bean
    @Primary
    @ConditionalOnProperty(name = "flexlb.monitor.enabled", havingValue = "false")
    public FlexMonitor noOpFlexMonitor() {
        return NoOpFlexMonitor.getInstance();
    }

    @Bean
    @ConditionalOnProperty(name = "flexlb.monitor.enabled", havingValue = "false")
    @ConditionalOnClass(MeterFilter.class)
    public MeterFilter denyAllMeterFilter() {
        return MeterFilter.deny();
    }
}
