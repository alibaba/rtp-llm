package org.flexlb.config;

import io.netty.channel.ChannelOption;
import org.springframework.boot.web.embedded.netty.NettyServerCustomizer;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import reactor.netty.resources.LoopResources;

@Configuration
public class NettyServerConfig {

    @Bean(destroyMethod = "dispose")
    public LoopResources loopResources(ConfigService configService) {
        int selectCount = Runtime.getRuntime().availableProcessors() * configService.loadBalanceConfig().getNettySelectThreadMultiplier();
        int workerCount = Runtime.getRuntime().availableProcessors() * configService.loadBalanceConfig().getNettyWorkerThreadMultiplier();
        return LoopResources.create(
                "reactor-netty-server",
                selectCount,
                workerCount,
                true
        );
    }

    @Bean
    public NettyServerCustomizer nettyServerMetricsCustomizer(ConfigService configService, LoopResources loopResources) {
        return httpServer -> httpServer
                .runOn(loopResources)
                .option(ChannelOption.SO_REUSEADDR, true)
                .option(ChannelOption.TCP_NODELAY, true);
    }
}
