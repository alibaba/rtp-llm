
package org.flexlb.config;

import io.micrometer.core.instrument.MeterRegistry;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
@ConditionalOnProperty(name = "flexlb.monitor.provider", havingValue = "micrometer")
public class MonitorClientConfig {

    /**
     * Create MicrometerFlexMonitor when the micrometer provider is selected and
     * kmonitor is not available (e.g. a test environment built with -P '!internal').
     *
     * <p>This bridges FlexLB business metrics to micrometer's MeterRegistry so they
     * are exposed via the {@code /prometheus} actuator endpoint without kmonitor.
     */
    @Bean
    @ConditionalOnProperty(name = "flexlb.monitor.enabled", havingValue = "true", matchIfMissing = true)
    @ConditionalOnMissingBean(FlexMonitor.class)
    @ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
    @ConditionalOnMissingClass("com.taobao.kmonitor.KMonitor")
    public FlexMonitor micrometerFlexMonitor(MeterRegistry meterRegistry) {
        log.info("Creating MicrometerFlexMonitor - bridging FlexMonitor to micrometer/Prometheus");
        return new MicrometerFlexMonitor(meterRegistry);
    }
}
