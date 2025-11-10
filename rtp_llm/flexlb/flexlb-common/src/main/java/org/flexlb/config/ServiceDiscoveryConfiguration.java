package org.flexlb.config;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.discovery.NoOpServiceDiscovery;
import org.flexlb.discovery.ServiceDiscovery;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * ServiceDiscoveryConfiguration - 服务发现默认配置
 *
 * @author saichen.sm
 */
@Slf4j
@Configuration
public class ServiceDiscoveryConfiguration {
    /**
     * 创建默认的ServiceDiscovery Bean
     * 当没有其他ServiceDiscovery实现时使用
     *
     * @return NoOpServiceDiscovery实例
     */
    @Bean
    @ConditionalOnMissingBean(ServiceDiscovery.class)
    public ServiceDiscovery serviceDiscovery() {
        log.info("Creating default NoOpServiceDiscovery (env-based discovery)");
        return NoOpServiceDiscovery.getInstance();
    }
}