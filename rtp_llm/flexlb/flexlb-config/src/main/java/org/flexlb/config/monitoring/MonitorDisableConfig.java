package org.flexlb.config.monitoring;

import io.micrometer.core.instrument.config.MeterFilter;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.NoOpFlexMonitor;
import org.springframework.boot.autoconfigure.condition.AnyNestedCondition;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Conditional;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.ConfigurationCondition.ConfigurationPhase;
import org.springframework.context.annotation.Primary;

/**
 * Installs no-op monitoring and denies metric publication when monitoring is disabled.
 */
@Configuration
public class MonitorDisableConfig {
    @Bean
    @Primary
    @Conditional(NoOpMonitorCondition.class)
    public FlexMonitor noOpFlexMonitor() {
        return NoOpFlexMonitor.getInstance();
    }

    @Bean
    @ConditionalOnProperty(name = "flexlb.monitor.enabled", havingValue = "false")
    @ConditionalOnClass(MeterFilter.class)
    public MeterFilter denyAllMeterFilter() {
        return MeterFilter.deny();
    }

    /**
     * Determines whether the monitoring configuration needs a no-op client.
     */
    static final class NoOpMonitorCondition extends AnyNestedCondition {

        NoOpMonitorCondition() {
            super(ConfigurationPhase.REGISTER_BEAN);
        }

        @ConditionalOnProperty(name = "flexlb.monitor.enabled", havingValue = "false")
        static class MonitoringDisabled {}

        @ConditionalOnProperty(
                name = "flexlb.monitor.provider",
                havingValue = "noop",
                matchIfMissing = true)
        static class NoOpProvider {}
    }
}
