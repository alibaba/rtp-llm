
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
     * 创建FlexMonitor实例
     * 使用NoOpFlexMonitor作为默认实现，如果需要启用监控，
     * 请添加internal_source/kmonitor-java依赖并设置环境变量FLEXLB_MONITOR_ENABLED=true
     */
    @Bean
    @ConditionalOnMissingBean(FlexMonitor.class)
    @ConditionalOnMissingClass("com.taobao.kmonitor.KMonitor")
    public FlexMonitor flexMonitor() {
        log.info("Creating default NoOpFlexMonitor - monitoring disabled");
        return NoOpFlexMonitor.getInstance();
    }
}