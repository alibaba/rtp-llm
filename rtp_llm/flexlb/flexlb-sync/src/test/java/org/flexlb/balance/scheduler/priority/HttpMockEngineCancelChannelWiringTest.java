package org.flexlb.balance.scheduler.priority;

import org.flexlb.engine.grpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

/**
 * Wiring contract for the mutually exclusive production and test Cancel
 * transports. Accepted Decode eviction itself is still gated only by
 * {@code AUTO_TPM_DECODE_ACCEPTED_EVICT_ENABLED} in the planner.
 */
class HttpMockEngineCancelChannelWiringTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withBean(EngineGrpcClient.class, () -> mock(EngineGrpcClient.class))
            .withUserConfiguration(EngineCancelChannelConfiguration.class);

    @Test
    void propertyAbsentSelectsProductionGrpcChannel() {
        runner.run(context -> {
            assertThat(context).hasSingleBean(EngineCancelChannel.class);
            assertThat(context.getBean(EngineCancelChannel.class))
                    .isInstanceOf(GrpcEngineCancelChannel.class);
            assertThat(context).doesNotHaveBean(UnsupportedEngineCancelChannel.class);
        });
    }

    @Test
    void nonBlankMockUrlSelectsHttpMockChannel() {
        runner.withPropertyValues(
                        "flexlb.test.mock-cancel-control-url=http://127.0.0.1:18089")
                .run(context -> {
                    assertThat(context).hasSingleBean(EngineCancelChannel.class);
                    assertThat(context.getBean(EngineCancelChannel.class))
                            .isInstanceOf(HttpMockEngineCancelChannel.class);
                });
    }

    @Test
    void configuredMockChannelSupportsEndpointsAndNormalizesTrailingSlash() {
        runner.withPropertyValues(
                        "flexlb.test.mock-cancel-control-url=http://127.0.0.1:18089/")
                .run(context -> {
                    EngineCancelChannel channel = context.getBean(EngineCancelChannel.class);
                    assertThat(channel).isInstanceOf(HttpMockEngineCancelChannel.class);
                    assertThat(channel.isSupported(null)).isTrue();
                });
    }

    @Test
    void emptyMockUrlSelectsProductionGrpcChannel() {
        runner.withPropertyValues("flexlb.test.mock-cancel-control-url=")
                .run(context -> {
                    assertThat(context).hasSingleBean(EngineCancelChannel.class);
                    assertThat(context.getBean(EngineCancelChannel.class))
                            .isInstanceOf(GrpcEngineCancelChannel.class);
                });
    }

    @Test
    void whitespaceMockUrlSelectsProductionGrpcChannel() {
        runner.withPropertyValues("flexlb.test.mock-cancel-control-url=   ")
                .run(context -> {
                    assertThat(context).hasSingleBean(EngineCancelChannel.class);
                    assertThat(context.getBean(EngineCancelChannel.class))
                            .isInstanceOf(GrpcEngineCancelChannel.class);
                });
    }
}
