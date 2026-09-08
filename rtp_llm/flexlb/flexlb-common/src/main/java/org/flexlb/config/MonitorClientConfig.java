
package org.flexlb.config;

import io.micrometer.core.instrument.MeterRegistry;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.flexlb.metric.NoOpFlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingClass;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Slf4j
@Configuration
public class MonitorClientConfig {

    /**
     * Create MicrometerFlexMonitor when MeterRegistry is on the classpath and
     * kmonitor is NOT available (e.g. 110 test environment built with -P '!internal').
     *
     * <p>This bridges FlexLB business metrics to micrometer's MeterRegistry so they
     * are exposed via the {@code /prometheus} actuator endpoint without kmonitor.
     */
    @Bean
    @ConditionalOnMissingBean(FlexMonitor.class)
    @ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
    @ConditionalOnMissingClass("com.taobao.kmonitor.KMonitor")
    public FlexMonitor micrometerFlexMonitor(MeterRegistry meterRegistry) {
        log.info("Creating MicrometerFlexMonitor - bridging FlexMonitor to micrometer/Prometheus");
        return new MicrometerFlexMonitor(meterRegistry);
    }

    /**
     * Fallback: create NoOpFlexMonitor when neither kmonitor nor micrometer is available.
     * To enable kmonitor-based monitoring, add internal_source/kmonitor-java dependency
     * and set environment variable FLEXLB_MONITOR_ENABLED=true
     */
    @Bean
    @ConditionalOnMissingBean(FlexMonitor.class)
    @ConditionalOnMissingClass("com.taobao.kmonitor.KMonitor")
    public FlexMonitor flexMonitor() {
        log.info("Creating default NoOpFlexMonitor - monitoring disabled");
        return NoOpFlexMonitor.getInstance();
    }
}