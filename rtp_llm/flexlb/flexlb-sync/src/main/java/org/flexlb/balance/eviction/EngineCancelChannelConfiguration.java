package org.flexlb.balance.eviction;

import org.flexlb.engine.grpc.EngineGrpcClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;

/**
 * Selects exactly one engine Cancel transport.
 *
 * <p>Production always uses gRPC. The HTTP implementation is reachable only
 * through the explicit test-only mock control URL. This transport choice is
 * deliberately separate from admission policy. Engine-owned Decode eviction
 * is enabled only when {@code preemption.allowedVictimStages} contains
 * {@code DECODE_ENGINE_OWNED}.
 */
@Configuration(proxyBeanMethods = false)
public class EngineCancelChannelConfiguration {

    @Bean
    public EngineCancelChannel engineCancelChannel(
            EngineGrpcClient engineGrpcClient,
            @Value("${flexlb.test.mock-cancel-control-url:}") String mockControlUrl) {
        if (StringUtils.hasText(mockControlUrl)) {
            return new HttpMockEngineCancelChannel(mockControlUrl.trim());
        }
        return new GrpcEngineCancelChannel(engineGrpcClient);
    }
}
