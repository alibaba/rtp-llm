package org.flexlb.config;

import org.springframework.boot.web.embedded.netty.NettyServerCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import reactor.netty.resources.LoopResources;

@Configuration
public class NettyServerConfig {

    @Bean(destroyMethod = "dispose")
    public LoopResources loopResources() {
        return LoopResources.create(
                "reactor-netty-server",
                Runtime.getRuntime().availableProcessors(),
                2 * Runtime.getRuntime().availableProcessors(),
                true
        );
    }

    @Bean
    public NettyServerCustomizer nettyServerMetricsCustomizer(LoopResources loopResources) {
        return httpServer -> httpServer
                .runOn(loopResources);
    }
}
